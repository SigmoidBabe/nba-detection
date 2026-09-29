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

TensorRT and PyCUDA require a compatible CUDA environment. The ONNX extra uses `onnxruntime`; install a suitable GPU runtime separately if you need ONNX inference on CUDA.

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

## TensorRT batch inference

Pass a list/tuple of BGR images (including different image sizes), or an NHWC
NumPy array. Batch input always returns a list of `DetectionResults` in input
order, including for a one-image batch; a single HWC image keeps the original
single-result return type. An empty batch returns an empty list.

```python
detector = Detector("model.trt")
images = [cv2.imread("first.jpg"), cv2.imread("second.jpg")]
results = detector.detect_n_seg(images, labels=["person"])
for image, result in zip(images, results):
    print(result.boxes.shape, result.image_shape)
```

The engine must expose an explicit NCHW float16/float32 input and raw YOLOv8
outputs, using linear device tensors. Optimization profile zero controls batch
limits; dynamic spatial dimensions use that profile's optimum height and width.
Requests larger than its maximum batch size are split into chunks. Fixed-batch
engines and profile minima are handled by repeating the last image in a short
chunk and discarding its extra predictions. An engine built for batch size one
still executes one image at a time; build an engine with a larger batch dimension
or profile to execute multiple images together.

Preprocessing normalizes and converts layout across each batch, retaining each
image's resize/padding metadata. Confidence and class filtering are vectorized
across the batch. NMS runs independently per image; box transforms and mask
reconstruction remain vectorized over detections, with bounded mask work chunks.
Boxes and masks are returned in each original image's coordinates. Detection
models retain stretch resizing, while segmentation models use letterboxing.

For incremental consumption, `detector.model.predict(images, stream=True)` yields
one result per image and executes one chunk at a time. `detect_n_seg` collects
these results into a list. Each TensorRT instance owns reusable pinned host/device
buffers and one execution context; use separate instances for concurrent calls.
ONNX batch inference is not supported by this wrapper.
