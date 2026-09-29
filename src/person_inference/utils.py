import cv2
import numpy as np


class DetectionResults:
    """One image of detections; boxes are [x1, y1, x2, y2, score, class, (id)]."""

    def __init__(self, boxes, names, masks=None, image_shape=None):
        boxes = np.asarray(boxes, dtype=np.float32)
        columns = boxes.shape[-1] if boxes.ndim == 2 else 6
        self.boxes = boxes.reshape(-1, columns)
        self.names = names
        self.masks = None if masks is None else np.asarray(masks, dtype=bool)
        self.image_shape = image_shape

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return DetectedObject(self, index)
        return DetectionResults(self.boxes[index], self.names,
                                None if self.masks is None else self.masks[index],
                                self.image_shape)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def xyxy(self):
        return self.boxes[:, :4]

    @property
    def scores(self):
        return self.boxes[:, 4]

    @property
    def class_ids(self):
        return self.boxes[:, 5].astype(np.intp)

    @property
    def areas(self):
        sizes = np.maximum(self.xyxy[:, 2:] - self.xyxy[:, :2], 0)
        return sizes[:, 0] * sizes[:, 1]

    def intersection_areas(self, other):
        """Pairwise box intersection areas, shape (len(self), len(other))."""
        left = np.maximum(self.xyxy[:, None, :2], other.xyxy[None, :, :2])
        right = np.minimum(self.xyxy[:, None, 2:], other.xyxy[None, :, 2:])
        size = np.maximum(right - left, 0)
        return size[..., 0] * size[..., 1]

    def intersects(self, other):
        """Pairwise box intersection flags, shape (len(self), len(other))."""
        return self.intersection_areas(other) > 0

    def image_mask(self, selected=None):
        """Union selected masks in original image coordinates, when requested."""
        if self.image_shape is None:
            raise ValueError("image_shape is required to scale masks")
        height, width = self.image_shape[:2]
        if self.masks is None or not len(self.masks):
            return np.zeros((height, width), dtype=np.uint8)
        masks = self.masks if selected is None else self.masks[selected]
        if not len(masks):
            return np.zeros((height, width), dtype=np.uint8)
        mask = np.any(masks, axis=0).astype(np.uint8)
        if mask.shape == (height, width):
            return mask
        gain = min(mask.shape[0] / height, mask.shape[1] / width)
        top = max(0, int(round((mask.shape[0] - height * gain) / 2 - 0.1)))
        left = max(0, int(round((mask.shape[1] - width * gain) / 2 - 0.1)))
        bottom = min(mask.shape[0], int(round(mask.shape[0] - top)))
        right = min(mask.shape[1], int(round(mask.shape[1] - left)))
        return cv2.resize(mask[top:bottom, left:right], (width, height),
                          interpolation=cv2.INTER_NEAREST)

    def plot(self, image):
        plotted = image.copy()
        if self.masks is not None and len(self.masks):
            mask = self.image_mask()
            plotted[mask != 0] = (0.5 * plotted[mask != 0] +
                                  0.5 * np.array([0, 255, 0])).astype(plotted.dtype)
        for obj in self:
            x1, y1, x2, y2 = obj.box
            cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(plotted, f"{obj.object_class} {obj.conf:.2f}",
                        (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
        return plotted


class DetectedObject:
    """Lightweight view of one detection row."""

    def __init__(self, results, index):
        self.results = results
        self.index = index

    @property
    def id(self):
        return int(self.results.boxes[self.index, 6]) if self.results.boxes.shape[1] == 7 else self.index

    @property
    def object_class(self):
        return self.results.names[int(self.results.boxes[self.index, 5])]

    @property
    def conf(self):
        return float(self.results.boxes[self.index, 4])

    @property
    def box(self):
        return self.results.boxes[self.index, :4].astype(np.intp).tolist()

    @property
    def area(self):
        return float(self.results.areas[self.index])

    @property
    def maskxy(self):
        if self.results.masks is None:
            return []
        mask = self.results.masks[self.index].astype(np.uint8)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return []
        points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if self.results.image_shape is not None and mask.shape != self.results.image_shape[:2]:
            height, width = self.results.image_shape[:2]
            gain = min(mask.shape[0] / height, mask.shape[1] / width)
            points -= ((mask.shape[1] - width * gain) / 2,
                       (mask.shape[0] - height * gain) / 2)
            points /= gain
            np.clip(points, (0, 0), (width, height), out=points)
        return points.tolist()


class BaseDetect:
    def createFormat(self, boxes, segments=None, *, image, masks=None):
        """Wrap postprocessed arrays without creating per-object results."""
        return [DetectionResults(boxes, self.classesModel, masks, image.shape)]

    def draw(self, orig_image, results):
        return orig_image, results[0].plot(orig_image)

class BatchUtils:
    """Batch transforms keep geometry explicit and numerical work vectorized."""

    letterbox = False

    @staticmethod
    def as_images(images):
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = list(images)
            else:
                raise ValueError("Expected an HWC image or NHWC image batch")
        elif isinstance(images, (list, tuple)):
            images = list(images)
        else:
            raise TypeError("Expected an image, a list/tuple of images, or an NHWC array")
        if any(not isinstance(im, np.ndarray) or im.ndim != 3 or
               im.shape[2] != 3 or min(im.shape[:2]) <= 0 for im in images):
            raise ValueError("Each image must be a nonempty HWC BGR array with three channels")
        return images

    def preprocess_batch(self, images):
        images = self.as_images(images)
        count = len(images)
        shapes = np.asarray([im.shape[:2] for im in images], dtype=np.intp).reshape(count, 2)
        target = np.asarray([self.model_width, self.model_height])
        ratios = target / shapes[:, ::-1]
        pads = np.zeros((count, 2), dtype=np.float32)
        if self.letterbox:
            ratios = np.repeat(ratios.min(axis=1, keepdims=True), 2, axis=1)
            sizes = np.maximum(np.rint(shapes[:, ::-1] * ratios).astype(np.intp), 1)
            pads = (target - sizes) / 2
        else:
            sizes = np.broadcast_to(target, (count, 2))
        # Resize is per image; color/layout conversion and normalization are batched.
        resized = np.empty((count, self.model_height, self.model_width, 3), dtype=self.ndtype)
        for index, image in enumerate(images):
            size = tuple(int(v) for v in sizes[index])
            frame = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            if self.letterbox:
                left, top = np.rint(pads[index] - 0.1).astype(np.intp)
                right, bottom = np.rint(pads[index] + 0.1).astype(np.intp)
                frame = cv2.copyMakeBorder(frame, int(top), int(bottom), int(left), int(right),
                                           cv2.BORDER_CONSTANT, value=(114, 114, 114))
            resized[index] = frame
        tensor = np.ascontiguousarray(resized[..., ::-1].transpose(0, 3, 1, 2))
        tensor /= np.asarray(255, dtype=self.ndtype)
        return tensor, shapes, ratios, pads

    def postprocess_batch(self, preds, shapes, ratios, pads,
                          conf_threshold, iou_threshold, classes=None):
        predictions = np.asarray(preds[0]).transpose(0, 2, 1)
        protos = np.asarray(preds[1]) if len(preds) == 2 else None
        count = len(shapes)
        if predictions.shape[0] != count or (protos is not None and protos.shape[0] != count):
            raise ValueError("Output batch size does not match preprocessing metadata")
        nm = protos.shape[1] if protos is not None else 0
        class_end = predictions.shape[2] - nm
        if class_end <= 4:
            raise ValueError("Expected raw YOLO output with at least one class score")
        scores = predictions[..., 4:class_end]
        class_ids = scores.argmax(axis=2)
        confidence = np.take_along_axis(scores, class_ids[..., None], axis=2)[..., 0]
        keep = confidence > conf_threshold
        if classes is not None:
            keep &= np.isin(class_ids, classes)
        # Gather once across the batch; NMS must not mix independent images.
        image_ids, anchor_ids = np.nonzero(keep)
        candidates = predictions[image_ids, anchor_ids].astype(np.float32, copy=False)
        confidence = confidence[image_ids, anchor_ids].astype(np.float32, copy=False)
        class_ids = class_ids[image_ids, anchor_ids]
        offsets = np.concatenate(([0], np.cumsum(np.bincount(image_ids, minlength=count))))
        results = []
        for index in range(count):
            selected = slice(offsets[index], offsets[index + 1])
            im0 = np.empty((*shapes[index], 0), dtype=np.uint8)
            results.append(UtilsSegment._postprocess_candidates(
                self, candidates[selected], confidence[selected], class_ids[selected],
                None if protos is None else protos[index], im0,
                ratios[index], *pads[index], conf_threshold, iou_threshold, False,
            ))
        return results


class UtilsDetect(BatchUtils):
    def __init__(self, width, height, dtype):
        self.model_height = height
        self.model_width = width
        self.ndtype = dtype

    def preprocess(self, image):
        self.image_height, self.image_width = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.model_width, self.model_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(self.ndtype)
        image_data = np.ascontiguousarray(image_data)
        return image_data

    def postprocess(self, output, conf_threshold, iou_threshold, classes=None):
        return UtilsSegment.postprocess(
            self, output, np.empty((self.image_height, self.image_width, 0)),
            (self.model_width / self.image_width, self.model_height / self.image_height),
            0, 0, conf_threshold, iou_threshold, classes=classes,
            return_segments=False,
        )

class BaseSegment(BaseDetect):
    """Shared array formatting for segmentation backends."""

class UtilsSegment(BatchUtils):
    letterbox = True

    def __init__(self, width, height, dtype):
        self.model_height = height
        self.model_width = width
        self.ndtype = dtype

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h,
                    conf_threshold, iou_threshold, nm=32, classes=None,
                    return_segments=True):
        """Decode one YOLOv8 image, keeping filtering and box transforms batched.

        Backends skip contour extraction; maskxy computes it on demand.
        The legacy (boxes, segments, masks) return signature is retained.
        """
        protos = preds[1][0] if len(preds) == 2 else None
        nm = protos.shape[0] if protos is not None else 0
        predictions = np.asarray(preds[0])[0].T
        class_end = predictions.shape[1] - nm
        scores = predictions[:, 4:class_end]
        class_ids = scores.argmax(axis=1)
        confidence = scores[np.arange(len(scores)), class_ids]
        keep = confidence > conf_threshold
        if classes is not None:
            keep &= np.isin(class_ids, classes)
        predictions = predictions[keep].astype(np.float32, copy=False)
        confidence = confidence[keep].astype(np.float32, copy=False)
        class_ids = class_ids[keep]
        return UtilsSegment._postprocess_candidates(
            self, predictions, confidence, class_ids, protos, im0, ratio, pad_w, pad_h,
            conf_threshold, iou_threshold, return_segments,
        )

    def _postprocess_candidates(self, predictions, confidence, class_ids, protos,
                                im0, ratio, pad_w, pad_h, conf_threshold,
                                iou_threshold, return_segments):
        nm = protos.shape[0] if protos is not None else 0
        if not len(predictions):
            masks = None if protos is None else np.empty((0, *im0.shape[:2]), dtype=bool)
            return np.empty((0, 6), dtype=np.float32), [], masks

        # OpenCV NMS expects top-left xywh, not center xywh. Offsets keep
        # objects belonging to different classes from suppressing each other.
        boxes = predictions[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] * 0.5
        extent = float(np.max(boxes[:, :2] + boxes[:, 2:]) - np.min(boxes[:, :2]) + 1)
        nms_boxes = boxes.copy()
        nms_boxes[:, :2] += class_ids[:, None] * extent
        indices = np.asarray(cv2.dnn.NMSBoxes(
            nms_boxes, confidence, conf_threshold, iou_threshold,
        ), dtype=np.intp).reshape(-1)
        boxes = boxes[indices]
        boxes[:, 2:] += boxes[:, :2]
        boxes -= np.asarray([pad_w, pad_h, pad_w, pad_h], dtype=np.float32)
        boxes /= np.asarray([ratio[0], ratio[1], ratio[0], ratio[1]], dtype=np.float32)
        np.clip(boxes, 0, [im0.shape[1], im0.shape[0], im0.shape[1], im0.shape[0]], out=boxes)
        result = np.column_stack((boxes, confidence[indices], class_ids[indices])).astype(np.float32)
        masks = None
        segments = []
        if protos is not None:
            masks = self.process_mask(protos, predictions[indices, -nm:], boxes, im0.shape)
            if return_segments:
                segments = self.masks2segments(masks)
        return result, segments, masks

    @staticmethod
    def masks2segments(masks):
        """
        It takes a list of masks(n,h,w) and returns a list of segments(n,xy) (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L750)

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        count = len(masks_in)
        height, width = im0_shape[:2]
        output = np.empty((count, height, width), dtype=bool)
        if not count:
            return output
        prototypes = np.asarray(protos, dtype=np.float32).reshape(c, -1)
        # Bound temporary full-resolution float buffers and OpenCV channels.
        chunk_size = max(1, min(128, (32 * 1024 * 1024) // (height * width * 4)))
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            logits = (np.asarray(masks_in[start:stop], dtype=np.float32) @ prototypes)
            logits = logits.reshape(-1, mh, mw).transpose(1, 2, 0)
            logits = self.scale_mask(np.ascontiguousarray(logits), im0_shape)
            # YOLOv8 prototype outputs are logits: sigmoid(logit) > .5
            # is equivalent to logit > 0, without computing a sigmoid.
            binary = logits.transpose(2, 0, 1) > 0
            output[start:stop] = self.crop_mask(binary, bboxes[start:stop])
        return output

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
