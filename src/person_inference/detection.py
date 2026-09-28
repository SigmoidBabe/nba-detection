import cv2
import numpy as np
import os
from shapely.geometry import Polygon, box
from .utils import DetectionResults, DetectedObject

class Detector:
    """
    A base class for implementing object segmentation and detection for Yolov8-seg

    Attributes:
        model_path: (required) path of the yolo8-seg pt model file
        isMax: (optional) only retrieve the detected objects with biggest area
        log: (optional) log object for verbose

    Methods:
        detect_n_seg: return a DetectionResults container with vectorized boxes and masks 
        remove_bg: remove background of chosen detected object
        crop: crop bounding box of chosen detected object
        get_label_idx: get index number of certain classses
        conv_seg: convert the result of raw inference to be formatted data
    
    Examples:
        VehicleDetector = Detector("path/to/yolo8-seg_model.pt")
        image = cv2.imread("path/to/image.jpg")
        result = VehicleDetector.detect_n_seg(image, labels=["car", "bike"], score_threshold=0.5)
    """
    def __init__(self, model_path:str):
        """
        initialize a new instance of Detector class.
        This constructor sets up the model based on the provided model path or name.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.

        Examples:
            >>> CarDetector = Detector("yolo8-seg.pt")
            >>> VehicleDetector = Detector("yolo8-seg.pt")
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        self.extension = model_path.split('.')[-1].lower()
        if self.extension == 'pt':
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        elif self.extension == 'onnx':
            from .onnx_handler.onnx_inference import InferenceOnnx
            self.model = InferenceOnnx(model_path)
        elif self.extension == 'trt':
            from .trt_handler.trt_inference import InferenceTrt
            self.model = InferenceTrt(model_path)
        basePath = os.path.dirname(model_path)
        nameModel,_ = os.path.splitext(os.path.basename(model_path))
        labelsPath = os.path.join(basePath, f"{nameModel}.txt")
        with open(labelsPath) as f:
            self.classes = f.read().split("\n")
    
    def get_label_idx(self, labels:list):
        if len(labels) > 0:
            clss = []
            for label in labels:
                if label in self.classes:
                    clss.append(self.classes.index(label))
                else:
                    raise ValueError(f'{label} Tidak Ditemukan')
        else:
            clss = None
        return clss

    def crop(self, image, box):
        xmin, ymin, xmax, ymax = box[:4]
        crop_face = image[ymin:ymax, xmin:xmax]
        return crop_face

    def detect_n_seg(self, image:np.ndarray, labels:list=[], score_threshold:float=0.5, stream=True):
        if len(labels) == 0: labels = self.classes
        clss = self.get_label_idx(labels)
        results = self.model.predict(image, conf=score_threshold, stream=stream, iou=0.4, classes=clss, verbose=False, save=False)
        if self.extension == 'pt':
            return self.convert_inference(results)
        return next(iter(results))
    
    def find_max_area(self, object_list, label):
        if isinstance(object_list, DetectionResults):
            class_ids = object_list.class_ids
            matches = np.flatnonzero([object_list.names[int(idx)] == label for idx in class_ids])
            return [object_list[int(matches[np.argmax(object_list.areas[matches])])]] if len(matches) else []
        max_obj = max((obj for obj in object_list if obj.object_class == label),
                      key=lambda obj: obj.area, default=None)
        return [max_obj] if max_obj else []

    def remove_bg(self, image, detected_objects, labels, main_label):
        if isinstance(detected_objects, DetectionResults):
            selected = np.array([detected_objects.names[int(idx)] in labels
                                 for idx in detected_objects.class_ids], dtype=bool)
            main = np.flatnonzero([detected_objects.names[int(idx)] == main_label
                                   for idx in detected_objects.class_ids])
            if not len(main):
                raise ValueError(f"No detection for {main_label}")
            mask = detected_objects.image_mask(selected)
            image_rm = cv2.bitwise_and(image, image, mask=mask)
            return self.crop(image_rm, detected_objects[int(main[-1])].box)

        mask = np.zeros_like(image)
        main_label_box = None
        for each in detected_objects:
            if each.object_class in labels and each.maskxy:
                points = np.asarray(each.maskxy, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], (255, 255, 255))
            if each.object_class == main_label:
                main_label_box = each.box
        if main_label_box is None:
            raise ValueError(f"No detection for {main_label}")
        return self.crop(cv2.bitwise_and(image, mask), main_label_box)

    def convert_inference(self, detections):
        """Convert PT output after predict; native backends already return this format."""
        converted = []
        for result in detections:
            if isinstance(result, DetectionResults):
                converted.append(result)
                continue
            boxes = result.boxes
            if boxes is None:
                data = np.empty((0, 6), dtype=np.float32)
            else:
                data = boxes.data.detach().cpu().numpy()
                if data.shape[1] == 7:
                    # Ultralytics track rows are xyxy, id, score, class.
                    data = data[:, [0, 1, 2, 3, 5, 6, 4]]
            masks = (result.masks.data.detach().cpu().numpy()
                     if result.masks is not None else None)
            converted.append(DetectionResults(data, result.names, masks,
                                              result.orig_shape))
        return converted[0] if len(converted) == 1 else converted

    def process_track(self, detections):
        return self.convert_inference(detections)

    def mask2polygon(self, maskxy):
        points = np.array(maskxy, np.int32)
        polygon = Polygon(points)
        return polygon
    
    def box2polygon(self, bbox):
        return box(bbox[0], bbox[1], bbox[2], bbox[3])
        
    def isIntersect(self, main:Polygon, child:Polygon):
        return main.intersects(child)
    
    def create_context(self):
        self.model.create_context()
        
