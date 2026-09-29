from ..utils import BaseSegment, UtilsSegment, UtilsDetect, DetectionResults
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Keep a CUDA context active for allocations and inference.
import os


class InferenceTrt(BaseSegment):
    """Explicit-batch YOLO TensorRT inference using optimization profile zero."""

    def __init__(self, modelPath):
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
        for name in names:
            if (self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE or
                    self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError("Only device tensors with linear storage are supported")
        self.stream = cuda.Stream()
        self.buffers = {}
        utils = UtilsSegment if self.segment else UtilsDetect
        self.utils = utils(self.opt_shape[3], self.opt_shape[2], self.dtypes[self.input_name])
        base, _ = os.path.splitext(modelPath)
        with open(base + ".txt") as f:
            self.classesModel = f.read().split("\n")

    def predict(self, image, conf=0.4, stream=True, classes=None, verbose=False,
                save=False, iou_threshold=0.7, nm=32, iou=None):
        images = self.utils.as_images(image)
        results = self._predict_batches(images, conf, iou_threshold if iou is None else iou, classes)
        return results if stream else list(results)

    def _predict_batches(self, images, conf, iou_threshold, classes):
        for start in range(0, len(images), self.max_batch):
            batch = images[start:start + self.max_batch]
            tensor, shapes, ratios, pads = self.utils.preprocess_batch(batch)
            count = len(batch)
            # Fixed batches and profile minima require padding the final chunk.
            # Padded predictions are discarded before postprocessing.
            if count < self.min_batch:
                padded = np.empty((self.min_batch, *tensor.shape[1:]), dtype=tensor.dtype)
                padded[:count] = tensor
                padded[count:] = tensor[-1]
                tensor = padded
            preds = [output[:count] for output in self.inf(tensor)]
            decoded = self.utils.postprocess_batch(preds, shapes, ratios, pads,
                                                   conf, iou_threshold, classes)
            for image, (boxes, _, masks) in zip(batch, decoded):
                yield DetectionResults(boxes, self.classesModel, masks, image.shape)

    def predict_seg(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        return self.predict(image, conf, False, classes, iou_threshold=iou_threshold, nm=nm)

    def predict_detect(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        return self.predict(image, conf, False, classes, iou_threshold=iou_threshold, nm=nm)

    def _buffer(self, name, shape):
        size = int(np.prod(shape))
        buffer = self.buffers.get(name)
        if buffer is None or buffer[0].size < size:
            host = cuda.pagelocked_empty(size, self.dtypes[name])
            device = cuda.mem_alloc(host.nbytes)
            buffer = self.buffers[name] = (host, device)
        host, device = buffer
        if not self.context.set_tensor_address(name, int(device)):
            raise RuntimeError(f"Could not bind TensorRT tensor {name}")
        return host[:size].reshape(shape), device

    def inf(self, image):
        """Execute one supported NCHW batch; returned arrays own their memory.

        An instance owns one execution context and is not safe for concurrent calls.
        """
        image = np.asarray(image, dtype=self.dtypes[self.input_name])
        shape = image.shape
        if len(shape) != 4 or any(s < lo or s > hi for s, lo, hi in
                                  zip(shape, self.min_shape, self.max_shape)):
            raise ValueError(f"Input shape {shape} is outside profile {self.min_shape}..{self.max_shape}")
        if not self.context.set_input_shape(self.input_name, shape):
            raise ValueError(f"TensorRT rejected input shape {shape}")
        output_shapes = [tuple(self.context.get_tensor_shape(n)) for n in self.output_names]
        if any(not s or min(s) <= 0 or s[0] != shape[0] for s in output_shapes):
            raise ValueError(f"Unsupported or unresolved TensorRT output shapes: {output_shapes}")
        host_input, device_input = self._buffer(self.input_name, shape)
        output_buffers = [self._buffer(n, s) for n, s in zip(self.output_names, output_shapes)]
        np.copyto(host_input, image)
        try:
            cuda.memcpy_htod_async(device_input, host_input, self.stream)
            if not self.context.execute_async_v3(stream_handle=self.stream.handle):
                raise RuntimeError("TensorRT execution failed")
            for host, device in output_buffers:
                cuda.memcpy_dtoh_async(host, device, self.stream)
        finally:
            self.stream.synchronize()
        return [host.copy() for host, _ in output_buffers]
