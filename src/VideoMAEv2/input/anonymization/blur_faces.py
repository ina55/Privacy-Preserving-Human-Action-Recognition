import cv2
import os

try:
    import mediapipe as mp
except Exception:
    mp = None


def _odd_kernel(kernel):
    kx, ky = kernel
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    return max(3, kx), max(3, ky)


def _expand_and_clip_box(x1, y1, x2, y2, width, height, padding_ratio):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(width, x2 + pad_x)
    ny2 = min(height, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def _mediapipe_boxes(detections, width, height, padding_ratio):
    boxes = []
    if not detections:
        return boxes
    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * width)
        y1 = int(bbox.ymin * height)
        x2 = x1 + int(bbox.width * width)
        y2 = y1 + int(bbox.height * height)
        x1, y1, x2, y2 = _expand_and_clip_box(x1, y1, x2, y2, width, height,
                                              padding_ratio)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _haar_boxes(gray_frame, cascade, width, height, padding_ratio):
    boxes = []
    detections = cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(20, 20),
    )
    for x, y, w, h in detections:
        x1, y1, x2, y2 = _expand_and_clip_box(x, y, x + w, y + h, width,
                                              height, padding_ratio)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _create_video_writer(output_path, fps, width, height):
    codecs = ['avc1', 'H264', 'mp4v']
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if writer.isOpened():
            print(f"Using output codec: {codec}")
            return writer
        writer.release()
    return None


def blur_faces_in_video(input_path,
                        output_path,
                        blur_kernel=(61, 61),
                        padding_ratio=0.25,
                        max_missed_frames=6,
                        min_detection_confidence=0.3):
    """
    Reads a video, detects faces using MediaPipe, applies a strong Gaussian blur 
    over the bounding box, and saves the result.
    """
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    face_detection_near = None
    face_detection_far = None
    if mp is not None and hasattr(mp, "solutions"):
        mp_face_detection = mp.solutions.face_detection
        face_detection_near = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        face_detection_far = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=max(0.2, min_detection_confidence - 0.05),
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or fps > 240:
        fps = 25.0

    out = _create_video_writer(output_path, fps, width, height)
    if out is None:
        print("Cannot create output video writer")
        cap.release()
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    blur_kernel = _odd_kernel(blur_kernel)

    print(f"Processing {input_path} -> {output_path}...")
    frame_count = 0
    detections_count = 0
    frames_blurred = 0
    previous_boxes = []
    missed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        boxes = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_detection_near is not None:
            results_near = face_detection_near.process(rgb_frame)
            boxes = _mediapipe_boxes(
                results_near.detections if results_near else None,
                width,
                height,
                padding_ratio,
            )

            if not boxes and face_detection_far is not None:
                results_far = face_detection_far.process(rgb_frame)
                boxes = _mediapipe_boxes(
                    results_far.detections if results_far else None,
                    width,
                    height,
                    padding_ratio,
                )

        if not boxes and not face_cascade.empty():
            boxes = _haar_boxes(gray_frame, face_cascade, width, height,
                                padding_ratio)

        if boxes:
            previous_boxes = boxes
            missed_frames = 0
            detections_count += len(boxes)
        elif previous_boxes and missed_frames < max_missed_frames:
            boxes = previous_boxes
            missed_frames += 1
        else:
            previous_boxes = []
            missed_frames = max_missed_frames

        for x1, y1, x2, y2 in boxes:
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_region,
                                                       blur_kernel, 0)
                frames_blurred += 1

        out.write(frame)

    cap.release()
    out.release()
    print(
        f"Done processing {frame_count} frames. Face detections: {detections_count}. "
        f"Blur applications: {frames_blurred}. Saved to {output_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        input_video = sys.argv[1]
        output_video = sys.argv[2]
        output_parent = os.path.dirname(output_video)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        blur_faces_in_video(input_video, output_video)
    else:
        os.makedirs("videos", exist_ok=True)
        blur_faces_in_video("videos/sample_face.mp4", "videos/sample_face_blurred.mp4")
