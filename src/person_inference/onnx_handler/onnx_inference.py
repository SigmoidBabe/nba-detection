from ..utils import BaseSegment, UtilsSegment
import os
import numpy as np
import onnxruntime as ort

class InferenceOnnx(BaseSegment):
    def __init__(self, modelPath):
        available = ort.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if name in available]
        self.session = ort.InferenceSession(modelPath, providers=providers)
        basePath = os.path.dirname(modelPath)
        nameModel,_ = os.path.splitext(os.path.basename(modelPath))
        labelsPath = os.path.join(basePath, f"{nameModel}.txt")
        with open(labelsPath) as f:
            self.classesModel = f.read().split("\n")

        ndtype = np.half if self.session.get_inputs()[0].type == "tensor(float16)" else np.single
        model_height, model_width = [x.shape for x in self.session.get_inputs()][0][-2:]
        self.utils = UtilsSegment(model_width, model_height, ndtype)

    def predict(self, image, conf=0.4, stream=True, classes=None, verbose=False,
                save=False, iou_threshold=0.7, nm=32, iou=None):
        im, ratio, (pad_w, pad_h) = self.utils.preprocess(image)
        preds = self.inf(im)
        boxes, segments, masks = self.utils.postprocess(
            preds, im0=image, ratio=ratio, pad_w=pad_w, pad_h=pad_h,
            conf_threshold=conf,
            iou_threshold=iou_threshold if iou is None else iou,
            nm=nm, classes=classes, return_segments=False,
        )
        results = self.createFormat(boxes, image=image, masks=masks)
        return iter(results) if stream else results

    def inf(self, image):
        return self.session.run(None, {self.session.get_inputs()[0].name: image})
