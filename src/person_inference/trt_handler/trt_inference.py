from ..utils import BaseSegment, BatchUtils, DetectionResults
import numpy as np
import tensorrt as trt
from numba import cuda
import os
from . import cuda_ops


class InferenceTrt(BaseSegment):
    """Explicit-batch YOLO TensorRT inference using optimization profile zero."""

    def __init__(self, modelPath):
        # Numba allocations and TensorRT execution share the CUDA primary context.
        cuda.current_context()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(modelPath, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise ValueError(f"Could not deserialize TensorRT engine: {modelPath}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create TensorRT execution context")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [name for name in names
                  if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        outputs = [name for name in names if name not in inputs]
        if len(inputs) != 1 or len(outputs) not in (1, 2):
            raise ValueError("Expected one NCHW input and YOLO detection/prototype outputs")
        self.input_name = inputs[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(self.input_shape) != 4:
            raise ValueError("Expected an explicit-batch NCHW image input")
        if -1 in self.input_shape:
            minimum, optimum, maximum = self.engine.get_tensor_profile_shape(self.input_name, 0)
        else:
            minimum = optimum = maximum = self.input_shape
        self.min_shape, self.opt_shape, self.max_shape = map(tuple, (minimum, optimum, maximum))
        if self.opt_shape[1] != 3 or min(self.min_shape) <= 0:
            raise ValueError("Expected positive NCHW profile dimensions with three channels")
        self.min_batch, self.max_batch = self.min_shape[0], self.max_shape[0]
        # Identify outputs by rank, independent of the engine's tensor ordering.
        detections = [n for n in outputs if len(self.engine.get_tensor_shape(n)) == 3]
        prototypes = [n for n in outputs if len(self.engine.get_tensor_shape(n)) == 4]
        if len(detections) != 1 or len(prototypes) != len(outputs) - 1:
            raise ValueError("Expected raw YOLO outputs [B, 4+classes+masks, anchors] and optional [B, masks, H, W]")
        self.output_names = detections + prototypes
        self.segment = bool(prototypes)
        self.dtypes = {name: np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                       for name in names}
        if self.dtypes[self.input_name] not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Expected float16 or float32 image input")
        if any(self.dtypes[name] not in (np.dtype(np.float16), np.dtype(np.float32))
               for name in self.output_names):
            raise ValueError("Expected float16 or float32 YOLO outputs")
        for name in names:
            if (self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
                    self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError("Only device tensors with linear storage are supported")
        self.stream = cuda.stream()
        self.buffers = {}
        self.host_frames = {}
        base, _ = os.path.splitext(modelPath)
        with open(base + ".txt") as f:
            self.classesModel = f.read().split("\n")

    def predict(self, image, conf=0.4, stream=True, classes=None, verbose=False,
                save=False, iou_threshold=0.7, nm=32, iou=None):
        images = BatchUtils.as_images(image)
        results = self._predict_batches(images, conf, iou_threshold if iou is None else iou, classes)
        return results if stream else list(results)

    def _predict_batches(self, images, conf, iou_threshold, classes):
        for start in range(0, len(images), self.max_batch):
            batch = images[start:start + self.max_batch]
            try:
                tensor, metadata = self._preprocess(batch)
                preds = self._execute_device(tensor)
                decoded = self._postprocess(preds, batch, metadata, conf,
                                            iou_threshold, classes)
            finally:
                # Includes failures: staging memory cannot be reused while in flight.
                self.stream.synchronize()
            yield from decoded

    def predict_seg(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        return self.predict(image, conf, False, classes, iou_threshold=iou_threshold, nm=nm)

    def predict_detect(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        return self.predict(image, conf, False, classes, iou_threshold=iou_threshold, nm=nm)

    def _buffer(self, name, shape, dtype):
        """Reuse flat device storage, returning a contiguous view of this shape."""
        size = int(np.prod(shape))
        dtype = np.dtype(dtype)
        buffer = self.buffers.get(name)
        if buffer is None or buffer.size < size or buffer.dtype != dtype:
            buffer = self.buffers[name] = cuda.device_array(size, dtype, stream=self.stream)
        return buffer[:size].reshape(shape)

    def _upload_frame(self, image, index):
        """Upload unprocessed HWC pixels before any CUDA image transform."""
        if image.dtype.kind not in "uif" or image.dtype.itemsize > 8:
            raise ValueError("Expected an image with a numeric pixel dtype")
        host = self.host_frames.get(index)
        if host is None or host.shape != image.shape or host.dtype != image.dtype:
            host = self.host_frames[index] = cuda.pinned_array(image.shape, image.dtype)
        # This is only a memory copy, including strided host images.
        np.copyto(host, image)
        raw = host.view(np.int16) if host.dtype == np.float16 else host
        frame = self._buffer(("frame", index), raw.shape, raw.dtype)
        frame.copy_to_device(raw, stream=self.stream)
        if host.dtype == np.float16:
            converted = self._buffer(("frame-float", index), image.shape, np.float32)
            cuda_ops.from_half[self._blocks(frame.size), cuda_ops.THREADS, self.stream](
                frame.reshape(frame.size), converted.reshape(converted.size))
            frame = converted
        return frame

    @staticmethod
    def _blocks(size):
        return (size + cuda_ops.THREADS - 1) // cuda_ops.THREADS

    def _preprocess(self, images):
        count = len(images)
        height, width = self.opt_shape[2:]
        shape = (max(count, self.min_batch), 3, height, width)
        tensor = self._buffer("preprocessed", shape, np.float32)
        # Geometry is calculated on CUDA; the host passes only image dimensions.
        metadata = self._buffer("geometry", (count, 8), np.float64)
        for index, image in enumerate(images):
            frame = self._upload_frame(image, index)
            cuda_ops.geometry[1, 1, self.stream](
                metadata, index, image.shape[0], image.shape[1], height, width, self.segment)
            cuda_ops.preprocess[self._blocks(height * width), cuda_ops.THREADS, self.stream](
                frame, tensor, metadata, index, image.dtype.kind in "ui")
        if count < shape[0]:
            size = (shape[0] - count) * 3 * height * width
            cuda_ops.repeat_last[self._blocks(size), cuda_ops.THREADS, self.stream](tensor, count)
        return tensor, metadata

    def _bind(self, name, device):
        address = int(device.__cuda_array_interface__["data"][0])
        if not self.context.set_tensor_address(name, address):
            raise RuntimeError(f"Could not bind TensorRT tensor {name}")

    def _execute_device(self, image):
        """Execute a device NCHW batch and leave all raw outputs on CUDA."""
        shape = tuple(image.shape)
        if len(shape) != 4 or any(s < lo or s > hi for s, lo, hi in
                                  zip(shape, self.min_shape, self.max_shape)):
            raise ValueError(f"Input shape {shape} is outside profile {self.min_shape}..{self.max_shape}")
        if not self.context.set_input_shape(self.input_name, shape):
            raise ValueError(f"TensorRT rejected input shape {shape}")
        output_shapes = [tuple(self.context.get_tensor_shape(n)) for n in self.output_names]
        if any(not s or min(s) <= 0 or s[0] != shape[0] for s in output_shapes):
            raise ValueError(f"Unsupported or unresolved TensorRT output shapes: {output_shapes}")
        if self.dtypes[self.input_name] == np.float16:
            # Represent half storage as int16 bits; Numba kernels compute in float32.
            device_input = self._buffer(("engine", self.input_name), shape, np.int16)
            cuda_ops.to_half[self._blocks(image.size), cuda_ops.THREADS, self.stream](
                image.reshape(image.size), device_input.reshape(device_input.size))
        else:
            device_input = image
        self._bind(self.input_name, device_input)
        outputs = []
        for name, output_shape in zip(self.output_names, output_shapes):
            dtype = np.int16 if self.dtypes[name] == np.float16 else np.float32
            device = self._buffer(("engine", name), output_shape, dtype)
            self._bind(name, device)
            outputs.append(device)
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        for index, (name, device) in enumerate(zip(self.output_names, outputs)):
            if self.dtypes[name] == np.float16:
                converted = self._buffer(("output-float", name), device.shape, np.float32)
                cuda_ops.from_half[self._blocks(device.size), cuda_ops.THREADS, self.stream](
                    device.reshape(device.size), converted.reshape(converted.size))
                outputs[index] = converted
        return outputs

    def _postprocess(self, preds, images, metadata, conf, iou_threshold, classes):
        predictions = preds[0]
        protos = preds[1] if self.segment else None
        count, anchors = len(images), predictions.shape[2]
        nm = protos.shape[1] if protos is not None else 0
        class_end = predictions.shape[1] - nm
        if class_end <= 4:
            raise ValueError("Expected raw YOLO output with at least one class score")
        allowed = np.asarray([] if classes is None else classes, dtype=np.int32).reshape(-1)
        device_classes = cuda.to_device(allowed, stream=self.stream)
        candidates = self._buffer("candidates", (count, anchors, 6), np.float32)
        scores = self._buffer("scores", (count, anchors), np.float32)
        selected = self._buffer("selected", (count, anchors), np.int32)
        counts = self._buffer("counts", (count,), np.int32)
        cuda_ops.decode[(self._blocks(anchors), count), (cuda_ops.THREADS, 1), self.stream](
            predictions, candidates, scores, device_classes, classes is not None, class_end, conf)
        cuda_ops.nms[count, cuda_ops.THREADS, self.stream](
            candidates, scores, selected, counts, iou_threshold)
        # Only counts cross to the host to size final result arrays.
        host_counts = cuda.pinned_array(count, np.int32)
        counts.copy_to_host(host_counts, stream=self.stream)
        self.stream.synchronize()
        results = []
        for index, (image, size) in enumerate(zip(images, host_counts)):
            size = int(size)
            height, width = image.shape[:2]
            if not size:
                masks = np.empty((0, height, width), dtype=bool) if self.segment else None
                results.append(DetectionResults(np.empty((0, 6), np.float32),
                                                self.classesModel, masks, image.shape))
                continue
            boxes = self._buffer("final-boxes", (size, 6), np.float32)
            cuda_ops.restore_boxes[self._blocks(size), cuda_ops.THREADS, self.stream](
                candidates, selected, metadata, index, boxes)
            masks = None
            if protos is not None:
                masks = self._buffer("final-masks", (size, height, width), np.bool_)
                mh, mw = protos.shape[2:]
                # Bound prototype-resolution scratch storage to 32 MiB.
                chunk_size = max(1, min(size, (32 * 1024 * 1024) // (mh * mw * 4)))
                for start in range(0, size, chunk_size):
                    chunk = min(chunk_size, size - start)
                    logits = self._buffer("mask-logits", (chunk, mh, mw), np.float32)
                    cuda_ops.mask_logits[(self._blocks(mh * mw), chunk), (cuda_ops.THREADS, 1), self.stream](
                        predictions, protos, selected, index, class_end, start, logits)
                    cuda_ops.restore_masks[(self._blocks(height * width), chunk), (cuda_ops.THREADS, 1), self.stream](
                        logits, boxes, start, masks)
            host_boxes = cuda.pinned_array(boxes.shape, np.float32)
            boxes.copy_to_host(host_boxes, stream=self.stream)
            host_masks = None
            if masks is not None:
                host_masks = cuda.pinned_array(masks.shape, np.bool_)
                masks.copy_to_host(host_masks, stream=self.stream)
            self.stream.synchronize()
            results.append(DetectionResults(host_boxes, self.classesModel, host_masks, image.shape))
        return results

    def inf(self, image):
        """Execute one supported NCHW batch; returned arrays own their memory.

        An instance owns one execution context and is not safe for concurrent calls.
        """
        # Compatibility entry point for callers supplying preprocessed host tensors.
        image = np.ascontiguousarray(image, dtype=np.float32)
        host = cuda.pinned_array(image.shape, np.float32)
        np.copyto(host, image)
        try:
            device = cuda.to_device(host, stream=self.stream)
            outputs = self._execute_device(device)
            results = [cuda.pinned_array(output.shape, np.float32) for output in outputs]
            for output, result in zip(outputs, results):
                output.copy_to_host(result, stream=self.stream)
        finally:
            self.stream.synchronize()
        return [result.astype(self.dtypes[name], copy=True)
                for name, result in zip(self.output_names, results)]
