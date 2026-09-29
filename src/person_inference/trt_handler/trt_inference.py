import os

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torchvision.ops import nms

from ..utils import BaseSegment


class InferenceTrt(BaseSegment):
    def __init__(self, modelPath):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(modelPath, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Strgiteam()

        self.input_name = self.engine.get_tensor_name(0)
        input_shape = tuple(self.engine.get_tensor_profile_shape(self.input_name, 0)[0])
        self.input_dtype = self._torch_dtype(self.input_name)
        self.model_height, self.model_width = input_shape[-2:]

        self.outputs = []
        for index in range(1, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[0])
            buffer = torch.empty(shape, dtype=self._torch_dtype(name), device="cuda")
            self.outputs.append(buffer)
            self.context.set_tensor_address(name, buffer.data_ptr())

        base_path = os.path.dirname(modelPath)
        model_name = os.path.splitext(os.path.basename(modelPath))[0]
        with open(os.path.join(base_path, f"{model_name}.txt")) as f:
            self.classesModel = f.read().split("\n")

    def _torch_dtype(self, name):
        dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
        return torch.from_numpy(np.empty((), dtype=dtype)).dtype

    def predict(self, image, conf=0.4, stream=True, classes=None, verbose=False,
                save=False, iou_threshold=0.7, nm=32, iou=None):
        threshold = iou_threshold if iou is None else iou
        with torch.cuda.stream(self.stream):
            tensor, ratio, pad = self._preprocess(image)
            predictions = self.inf(tensor)
            boxes, masks = self._postprocess(
                predictions, image.shape, ratio, pad, conf, threshold, classes
            )
            # The public result API exposes NumPy arrays, so only final results
            # cross back to CPU memory.
            boxes = boxes.cpu().numpy()
            masks = masks.cpu().numpy() if masks is not None else None
        results = self.createFormat(boxes, image=image, masks=masks)
        return iter(results) if stream else results

    def _preprocess(self, image):
        height, width = image.shape[:2]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device="cuda")
        tensor = tensor.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0)

        if len(self.outputs) == 1:
            ratio = self.model_width / width, self.model_height / height
            pad = 0.0, 0.0
            size = self.model_height, self.model_width
        else:
            gain = min(self.model_height / height, self.model_width / width)
            ratio = gain, gain
            size = round(height * gain), round(width * gain)
            pad_w = (self.model_width - size[1]) / 2
            pad_h = (self.model_height - size[0]) / 2
            pad = pad_w, pad_h

        tensor = F.interpolate(tensor.float(), size=size, mode="bilinear", align_corners=False)
        if len(self.outputs) > 1:
            left, top = round(pad[0] - 0.1), round(pad[1] - 0.1)
            right = self.model_width - size[1] - left
            bottom = self.model_height - size[0] - top
            tensor = F.pad(tensor, (left, right, top, bottom), value=114.0)
        return (tensor / 255.0).to(self.input_dtype).contiguous(), ratio, pad

    def inf(self, image):
        self.context.set_tensor_address(self.input_name, image.data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        return self.outputs

    def _postprocess(self, outputs, image_shape, ratio, pad, conf, iou, classes):
        prototypes = outputs[1][0] if len(outputs) > 1 else None
        mask_dim = prototypes.shape[0] if prototypes is not None else 0
        predictions = outputs[0][0].T
        scores = predictions[:, 4:predictions.shape[1] - mask_dim]
        confidence, class_ids = scores.max(dim=1)
        keep = confidence > conf
        if classes is not None:
            keep &= torch.isin(class_ids, torch.as_tensor(classes, device=class_ids.device))
        predictions = predictions[keep].float()
        confidence = confidence[keep].float()
        class_ids = class_ids[keep]

        if predictions.shape[0] == 0:
            empty_boxes = torch.empty((0, 6), dtype=torch.float32, device="cuda")
            empty_masks = (torch.empty((0, *image_shape[:2]), dtype=torch.bool, device="cuda")
                           if prototypes is not None else None)
            return empty_boxes, empty_masks

        boxes = predictions[:, :4].clone()
        boxes[:, :2] -= boxes[:, 2:] * 0.5
        extent = (boxes[:, :2] + boxes[:, 2:]).max() - boxes[:, :2].min() + 1
        nms_boxes = boxes.clone()
        nms_boxes[:, :2] += class_ids[:, None] * extent
        nms_boxes[:, 2:] += nms_boxes[:, :2]
        selected = nms(nms_boxes, confidence, iou)

        boxes = boxes[selected]
        boxes[:, 2:] += boxes[:, :2]
        boxes -= boxes.new_tensor((pad[0], pad[1], pad[0], pad[1]))
        boxes /= boxes.new_tensor((ratio[0], ratio[1], ratio[0], ratio[1]))
        boxes[:, 0].clamp_(0, image_shape[1])
        boxes[:, 2].clamp_(0, image_shape[1])
        boxes[:, 1].clamp_(0, image_shape[0])
        boxes[:, 3].clamp_(0, image_shape[0])
        result = torch.cat((boxes, confidence[selected, None], class_ids[selected, None]), dim=1)

        masks = None
        if prototypes is not None:
            masks = self._process_masks(prototypes, predictions[selected, -mask_dim:],
                                        boxes, image_shape)
        return result, masks

    @staticmethod
    def _process_masks(prototypes, coefficients, boxes, image_shape):
        count = coefficients.shape[0]
        height, width = image_shape[:2]
        output = torch.empty((count, height, width), dtype=torch.bool, device="cuda")
        if not count:
            return output

        channels, mask_height, mask_width = prototypes.shape
        flattened = prototypes.float().reshape(channels, -1)
        chunk_size = max(1, min(128, (32 * 1024 * 1024) // (height * width * 4)))
        gain = min(mask_height / height, mask_width / width)
        pad_w = (mask_width - width * gain) / 2
        pad_h = (mask_height - height * gain) / 2
        top, left = round(pad_h - 0.1), round(pad_w - 0.1)
        bottom, right = round(mask_height - pad_h + 0.1), round(mask_width - pad_w + 0.1)
        rows = torch.arange(height, device="cuda")[None, :, None]
        cols = torch.arange(width, device="cuda")[None, None, :]

        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            logits = coefficients[start:stop].float() @ flattened
            logits = logits.reshape(-1, mask_height, mask_width)
            logits = F.interpolate(logits[:, None, top:bottom, left:right],
                                   size=(height, width), mode="bilinear",
                                   align_corners=False)[:, 0]
            x1, y1, x2, y2 = boxes[start:stop].unbind(dim=1)
            inside = ((cols >= x1[:, None, None]) & (cols < x2[:, None, None]) &
                      (rows >= y1[:, None, None]) & (rows < y2[:, None, None]))
            output[start:stop] = (logits > 0) & inside
        return output
