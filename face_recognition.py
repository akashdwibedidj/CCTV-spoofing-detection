import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import time

model_path = 'F:\\Projects\\CCTV_Spoofing_Detector\\models\\blaze_face_short_range.tflite'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_mp: int):
    global latest_result
    latest_result = result

options = FaceDetectorOptions(
    base_options = BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
    )
with FaceDetector.create_from_options(options) as detector:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Can't receive frame(stream end?). Eciting ...")
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int((time.time()-start_time) * 1000)
        detector.detect_async(mp_image, frame_timestamp_ms)

        if latest_result is not None:
            h, w = frame.shape[:2]
            for detection in latest_result.detections:
                bbox = detection.bounding_box
                cv.rectangle(
                    frame,
                    (bbox.origin_x, bbox.origin_y),
                    (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                    (0, 255, 0), 2
                )
                score = detection.categories[0].score
                label = f"{score:.2f}"
                cv.putText(
                    frame, label,
                    (bbox.origin_x, max(0, bbox.origin_y - 10)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                for kp in detection.keypoints:
                    x_px = int(kp.x * w)
                    y_px = int(kp.y * h)
                    cv.circle(frame, (x_px, y_px), 2, (0, 0, 255), -1)
                
        
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        
cap.release()
cv.destroyAllWindows()