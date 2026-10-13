"""Device-only image transforms and raw YOLO decoding for TensorRT."""

import math

from numba import cuda, float32, int32
from numba.cuda import libdevice


THREADS = 256


@cuda.jit
def to_half(source, destination):
    index = cuda.grid(1)
    if index < source.size:
        destination[index] = libdevice.float2half_rn(float32(source[index]))


@cuda.jit
def from_half(source, destination):
    index = cuda.grid(1)
    if index < source.size:
        destination[index] = libdevice.half2float(source[index])


@cuda.jit
def geometry(metadata, index, height, width, target_h, target_w, letterbox):
    if cuda.grid(1) != 0:
        return
    rx, ry = target_w / width, target_h / height
    resized_w, resized_h = target_w, target_h
    px, py = 0.0, 0.0
    if letterbox:
        rx = min(rx, ry)
        ry = rx
        resized_w = max(1, int(round(width * rx)))
        resized_h = max(1, int(round(height * ry)))
        px = (target_w - resized_w) / 2.0
        py = (target_h - resized_h) / 2.0
    metadata[index, 0] = height
    metadata[index, 1] = width
    metadata[index, 2] = rx
    metadata[index, 3] = ry
    metadata[index, 4] = px
    metadata[index, 5] = py
    metadata[index, 6] = resized_w
    metadata[index, 7] = resized_h


@cuda.jit
def preprocess(frame, tensor, metadata, index, rounded_pixels):
    """Fuse bilinear resize, padding, BGR->RGB, HWC->NCHW and normalization."""
    pixel = cuda.grid(1)
    target_h, target_w = tensor.shape[2], tensor.shape[3]
    if pixel >= target_h * target_w:
        return
    y, x = pixel // target_w, pixel % target_w
    resized_w = int(metadata[index, 6])
    resized_h = int(metadata[index, 7])
    left = int(round(metadata[index, 4] - 0.1))
    top = int(round(metadata[index, 5] - 0.1))
    inside = left <= x < left + resized_w and top <= y < top + resized_h
    for channel in range(3):
        value = float32(114.0)
        if inside:
            sx = max(0.0, min(frame.shape[1] - 1.0,
                             (x - left + 0.5) * frame.shape[1] / resized_w - 0.5))
            sy = max(0.0, min(frame.shape[0] - 1.0,
                             (y - top + 0.5) * frame.shape[0] / resized_h - 0.5))
            x0, y0 = int(math.floor(sx)), int(math.floor(sy))
            x1, y1 = min(x0 + 1, frame.shape[1] - 1), min(y0 + 1, frame.shape[0] - 1)
            wx, wy = float32(sx - x0), float32(sy - y0)
            c = 2 - channel
            value = ((1 - wy) * ((1 - wx) * float32(frame[y0, x0, c]) +
                                wx * float32(frame[y0, x1, c])) +
                     wy * ((1 - wx) * float32(frame[y1, x0, c]) +
                           wx * float32(frame[y1, x1, c])))
            if rounded_pixels:
                value = float32(round(value))
        tensor[index, channel, y, x] = value / float32(255.0)


@cuda.jit
def repeat_last(tensor, count):
    index = cuda.grid(1)
    per_image = tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
    if index < (tensor.shape[0] - count) * per_image:
        flat = tensor.reshape(tensor.size)
        flat[count * per_image + index] = flat[(count - 1) * per_image + index % per_image]


@cuda.jit
def decode(predictions, candidates, scores, allowed_classes, filter_classes,
           class_end, threshold):
    anchor, image = cuda.grid(2)
    if image >= scores.shape[0] or anchor >= scores.shape[1]:
        return
    best_class = 0
    best_score = predictions[image, 4, anchor]
    for c in range(5, class_end):
        score = predictions[image, c, anchor]
        if score > best_score:
            best_score, best_class = score, c - 4
    allowed = not filter_classes
    for i in range(allowed_classes.size):
        if allowed_classes[i] == best_class:
            allowed = True
    scores[image, anchor] = best_score if allowed and best_score > threshold else -1.0
    cx, cy = predictions[image, 0, anchor], predictions[image, 1, anchor]
    w, h = predictions[image, 2, anchor], predictions[image, 3, anchor]
    candidates[image, anchor, 0] = cx - w * 0.5
    candidates[image, anchor, 1] = cy - h * 0.5
    candidates[image, anchor, 2] = cx + w * 0.5
    candidates[image, anchor, 3] = cy + h * 0.5
    candidates[image, anchor, 4] = best_score
    candidates[image, anchor, 5] = best_class


@cuda.jit
def nms(candidates, scores, selected, counts, threshold):
    """One cooperative block per image; greedy, class-aware, descending-score NMS."""
    image, thread = cuda.blockIdx.x, cuda.threadIdx.x
    best_scores = cuda.shared.array(THREADS, float32)
    best_ids = cuda.shared.array(THREADS, int32)
    count = 0
    while True:
        score, anchor = float32(-1.0), -1
        for i in range(thread, scores.shape[1], THREADS):
            if scores[image, i] > score:
                score, anchor = scores[image, i], i
        best_scores[thread], best_ids[thread] = score, anchor
        cuda.syncthreads()
        stride = THREADS // 2
        while stride:
            if thread < stride:
                other_score, other_id = best_scores[thread + stride], best_ids[thread + stride]
                if (other_score > best_scores[thread] or
                        (other_score == best_scores[thread] and other_id >= 0 and
                         (best_ids[thread] < 0 or other_id < best_ids[thread]))):
                    best_scores[thread], best_ids[thread] = other_score, other_id
            cuda.syncthreads()
            stride //= 2
        winner = best_ids[0]
        if winner < 0:
            break
        if thread == 0:
            selected[image, count] = winner
        x1, y1 = candidates[image, winner, 0], candidates[image, winner, 1]
        x2, y2 = candidates[image, winner, 2], candidates[image, winner, 3]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cls = candidates[image, winner, 5]
        for i in range(thread, scores.shape[1], THREADS):
            if i == winner:
                scores[image, i] = -1.0
            elif scores[image, i] >= 0 and candidates[image, i, 5] == cls:
                bx1, by1 = candidates[image, i, 0], candidates[image, i, 1]
                bx2, by2 = candidates[image, i, 2], candidates[image, i, 3]
                intersection = max(0.0, min(x2, bx2) - max(x1, bx1)) * max(
                    0.0, min(y2, by2) - max(y1, by1))
                union = area + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
                if union > 0 and intersection / union > threshold:
                    scores[image, i] = -1.0
        count += 1
        cuda.syncthreads()
    if thread == 0:
        counts[image] = count


@cuda.jit
def restore_boxes(candidates, selected, metadata, image, boxes):
    index = cuda.grid(1)
    if index >= boxes.shape[0]:
        return
    anchor = selected[image, index]
    for axis in range(4):
        dimension = axis % 2
        value = (candidates[image, anchor, axis] - metadata[image, 4 + dimension]) / metadata[image, 2 + dimension]
        limit = metadata[image, 1 - dimension]
        boxes[index, axis] = min(limit, max(0.0, value))
    boxes[index, 4] = candidates[image, anchor, 4]
    boxes[index, 5] = candidates[image, anchor, 5]


@cuda.jit
def mask_logits(predictions, protos, selected, image, class_end, start, logits):
    pixel, detection = cuda.grid(2)
    mh, mw = protos.shape[2], protos.shape[3]
    if pixel >= mh * mw or detection >= logits.shape[0]:
        return
    anchor = selected[image, start + detection]
    value = float32(0.0)
    for channel in range(protos.shape[1]):
        value += predictions[image, class_end + channel, anchor] * protos[image, channel, pixel // mw, pixel % mw]
    logits[detection, pixel // mw, pixel % mw] = value


@cuda.jit
def restore_masks(logits, boxes, start, masks):
    pixel, detection = cuda.grid(2)
    height, width = masks.shape[1], masks.shape[2]
    if pixel >= height * width or detection >= logits.shape[0]:
        return
    y, x = pixel // width, pixel % width
    index = start + detection
    inside = (boxes[index, 0] <= x < boxes[index, 2] and
              boxes[index, 1] <= y < boxes[index, 3])
    value = float32(0.0)
    if inside:
        mh, mw = logits.shape[1], logits.shape[2]
        gain = min(mh / height, mw / width)
        px, py = (mw - width * gain) / 2, (mh - height * gain) / 2
        left, top = int(round(px - 0.1)), int(round(py - 0.1))
        right, bottom = int(round(mw - px + 0.1)), int(round(mh - py + 0.1))
        left, top = min(mw - 1, max(0, left)), min(mh - 1, max(0, top))
        right, bottom = min(mw, max(left + 1, right)), min(mh, max(top + 1, bottom))
        sx = max(0.0, min(right - left - 1.0, (x + 0.5) * (right - left) / width - 0.5)) + left
        sy = max(0.0, min(bottom - top - 1.0, (y + 0.5) * (bottom - top) / height - 0.5)) + top
        x0, y0 = int(math.floor(sx)), int(math.floor(sy))
        x1, y1 = min(x0 + 1, right - 1), min(y0 + 1, bottom - 1)
        wx, wy = float32(sx - x0), float32(sy - y0)
        value = ((1 - wy) * ((1 - wx) * logits[detection, y0, x0] + wx * logits[detection, y0, x1]) +
                 wy * ((1 - wx) * logits[detection, y1, x0] + wx * logits[detection, y1, x1]))
    masks[index, y, x] = inside and value > 0
