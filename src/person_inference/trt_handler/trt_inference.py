from ..utils import BaseSegment, UtilsSegment, BaseDetect, UtilsDetect
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Keep a CUDA context active for TensorRT allocations and inference.
import os

class InferenceTrt(BaseSegment):
    def __init__(self, modelPath):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(modelPath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        self.context = engine.create_execution_context()
        
        input_name = engine.get_tensor_name(0)
        input_shape = engine.get_tensor_profile_shape(input_name, 0)[0]
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
        size_input = trt.volume(input_shape)
        host_mem_input = cuda.pagelocked_empty(size_input, input_dtype)
        self.d_input = cuda.mem_alloc(host_mem_input.nbytes)
        # h_input = cuda.mem_alloc(host_mem_input.nbytes)

        output0_name = engine.get_tensor_name(1)
        self.output0_shape = engine.get_tensor_profile_shape(output0_name, 0)[0]
        self.output0_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output0_name)))
        size_output0 = trt.volume(self.output0_shape)
        host_mem_output0 = cuda.pagelocked_empty(size_output0, self.output0_dtype)
        self.d_output0 = cuda.mem_alloc(host_mem_output0.nbytes)
        # h_output0 = cuda.mem_alloc(host_mem_output0.nbytes)

        output1_name = engine.get_tensor_name(2) if engine.num_io_tensors > 2 else None
        if output1_name:
            print("Segmentation Model")
            self.output1_shape = engine.get_tensor_profile_shape(output1_name, 0)[0]
            self.output1_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output1_name)))
            size_output1 = trt.volume(self.output1_shape)
            host_mem_output1 = cuda.pagelocked_empty(size_output1, self.output1_dtype)
            self.d_output1 = cuda.mem_alloc(host_mem_output1.nbytes)
            self.bindings = [int(self.d_input), int(self.d_output0), int(self.d_output1)]
        # h_output1 = cuda.mem_alloc(host_mem_output1.nbytes)
        else:
            print("Detection Model")
            self.bindings = [int(self.d_input), int(self.d_output0)]
        self.engine = engine
        self.stream = cuda.Stream()

        basePath = os.path.dirname(modelPath)
        nameModel,_ = os.path.splitext(os.path.basename(modelPath))
        labelsPath = os.path.join(basePath, f"{nameModel}.txt")
        with open(labelsPath) as f:
            self.classesModel = f.read().split("\n")

        self.context.set_tensor_address(input_name, int(self.d_input))
        self.context.set_tensor_address(output0_name, int(self.d_output0))
        if output1_name:
            self.context.set_tensor_address(output1_name, int(self.d_output1))
            self.utils = UtilsSegment(input_shape[3], input_shape[2], input_dtype)
        else:
            self.utils = UtilsDetect(input_shape[3], input_shape[2], input_dtype)
        

    def predict(self, image, conf=0.4, stream=True, classes=None, verbose=False,
                save=False, iou_threshold=0.7, nm=32, iou=None):
        threshold = iou_threshold if iou is None else iou
        if len(self.bindings) == 3:
            results = self.predict_seg(image, conf, threshold, nm, classes)
        else:
            results = self.predict_detect(image, conf, threshold, nm, classes)
        return iter(results) if stream else results

    def predict_seg(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        im, ratio, (pad_w, pad_h) = self.utils.preprocess(image)
        preds = self.inf(im)
        boxes, segments, masks = self.utils.postprocess(
            preds, im0=image, ratio=ratio, pad_w=pad_w, pad_h=pad_h,
            conf_threshold=conf, iou_threshold=iou_threshold,
            nm=nm, classes=classes, return_segments=False,
        )
        return self.createFormat(boxes, image=image, masks=masks)

    def predict_detect(self, image, conf=0.4, iou_threshold=0.7, nm=32, classes=None):
        preds = self.inf(self.utils.preprocess(image))
        boxes, segments, masks = self.utils.postprocess(
            preds, conf_threshold=conf, iou_threshold=iou_threshold, classes=classes,
        )
        return self.createFormat(boxes, image=image)

    def inf(self, image):
        cuda.memcpy_htod_async(self.d_input, image, self.stream)
        # self.context.execute_v2(self.bindings)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        if len(self.bindings)==3:
            output0 = np.empty(self.output0_shape, dtype=self.output0_dtype)
            output1 = np.empty(self.output1_shape, dtype=self.output1_dtype)
            cuda.memcpy_dtoh_async(output0, self.d_output0, self.stream)
            cuda.memcpy_dtoh_async(output1, self.d_output1, self.stream)
            self.stream.synchronize()
            return [output0, output1]
        else:
            output0 = np.empty(self.output0_shape, dtype=self.output0_dtype)
            cuda.memcpy_dtoh_async(output0, self.d_output0, self.stream)
            self.stream.synchronize()
            return [output0]
