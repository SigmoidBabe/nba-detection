# person-inference

Image detection and segmentation with optional Ultralytics (`.pt`), ONNX (`.onnx`), and TensorRT (`.trt`) backends.

## Install

Choose the backend you need. The base package contains shared NumPy, OpenCV, and Shapely code; backend libraries are installed only through the selected extra.

```bash
pip install '.[ultralytics]'
pip install '.[onnx]'
pip install '.[tensorrt]'
```

After pushing this directory to GitHub, install directly from the repository. Replace `OWNER/REPO` with its actual location:

```bash
pip install 'person-inference[ultralytics] @ git+https://github.com/OWNER/REPO.git'
pip install 'person-inference[onnx] @ git+https://github.com/OWNER/REPO.git'
pip install 'person-inference[tensorrt] @ git+https://github.com/OWNER/REPO.git'
```

TensorRT, PyTorch, and TorchVision require a compatible CUDA environment. The ONNX extra uses `onnxruntime`; install a suitable GPU runtime separately if you need ONNX inference on CUDA.

## Inference

Place a text file with the same stem as the model next to it, with one class name per line (for example, `model.txt` for `model.trt`).

```python
import cv2
from person_inference import Detector

image = cv2.imread("image.jpg")
if image is None:
    raise FileNotFoundError("image.jpg")

results = Detector("model.trt").detect_n_seg(image)
print([obj.object_class for obj in results])
cv2.imwrite("output.jpg", results.plot(image))
```

`results.boxes` is a NumPy array with columns `x1, y1, x2, y2, score, class`, plus an optional tracking ID. `results.masks` is a batched boolean array when segmentation masks are available. `results.intersects(other)` and `results.intersection_areas(other)` calculate pairwise box intersections.
