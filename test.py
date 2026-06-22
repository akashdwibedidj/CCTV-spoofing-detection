import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import time
import tensorflow as tf
import json
import os
from collections import deque

# --- File Paths ---
model_path = r'F:\Projects\CCTV_Spoofing_Detector\models\blaze_face_short_range.tflite'
spoof_model_path = r'F:\Projects\CCTV_Spoofing_Detector\models\new_processed_dataset_trainable_all-true_1_40epoch.keras'
facenet_model_path = r'F:\Projects\CCTV_Spoofing_Detector\models\facenet.tflite'
DB_JSON_PATH = r'F:\Projects\CCTV_Spoofing_Detector\database\face_database.json'

# --- Configurations ---
SPOOF_MARGIN_RATIO = 0.25
SPOOF_INPUT_SIZE = (224, 224)
SPOOF_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 0.6
FACENET_INPUT_SIZE = 160

# --- Temporal Smoothing Parameters ---
BUFFER_DURATION_SEC = 3.0    # Window size to confirm a SPOOF
SPOOF_RATIO_TRIGGER = 0.60    # 60% of frames must be spoof to lock red
INSTANT_REAL_CONSECUTIVE_FRAMES = 3  # 3 back-to-back real frames trigger instant reset

# --- Load Models ---
spoof_model = tf.keras.models.load_model(spoof_model_path)
facenet_interpreter = tf.lite.Interpreter(model_path=facenet_model_path)
facenet_interpreter.allocate_tensors()
facenet_input_details = facenet_interpreter.get_input_details()
facenet_output_details = facenet_interpreter.get_output_details()

# --- Multi-Face Tracker Databases ---
known_face_db = {}
# Structure: { face_key: deque([(timestamp, score), ...]) }
face_temporal_buffers = {}
# Structure: { face_key: int_consecutive_real_count }
face_real_counters = {}

# --- Load JSON Face Database ---
if os.path.exists(DB_JSON_PATH):
    try:
        with open(DB_JSON_PATH, 'r') as f:
            raw_data = json.load(f)
            known_face_db = {name: np.array(emb, dtype=np.float32) for name, emb in raw_data.items()}
        print(f"Loaded {len(known_face_db)} users from permanent database.")
    except Exception as e:
        print(f"Error loading database: {e}")
else:
    os.makedirs(os.path.dirname(DB_JSON_PATH), exist_ok=True)


def save_database_to_json():
    serializable_db = {name: emb.tolist() for name, emb in known_face_db.items()}
    with open(DB_JSON_PATH, 'w') as f:
        json.dump(serializable_db, f, indent=4)
    print("Database permanently synced to JSON.")


def get_embedding(aligned_face_rgb):
    face_input = (aligned_face_rgb.astype(np.float32) - 127.5) / 128.0
    face_input = np.expand_dims(face_input, axis=0)
    facenet_interpreter.set_tensor(facenet_input_details[0]['index'], face_input)
    facenet_interpreter.invoke()
    embedding = facenet_interpreter.get_tensor(facenet_output_details[0]['index'])
    embedding = np.squeeze(embedding)
    return embedding / np.linalg.norm(embedding)


def recognize_face(embedding):
    if not known_face_db:
        return "Unknown"
    best_match, min_distance = "Unknown", float('inf')
    for name, known_emb in known_face_db.items():
        distance = 1.0 - np.dot(embedding, known_emb)
        if distance < min_distance:
            min_distance = distance
            best_match = name
    if min_distance < RECOGNITION_THRESHOLD:
        return best_match
    return "Unknown"


def align_face(image, left_eye, right_eye, desired_face_width=160):
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    dist = np.sqrt(dx**2 + dy**2)
    desired_dist = 0.3 * desired_face_width
    scale = desired_dist / dist if dist > 0 else 1.0
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    M = cv.getRotationMatrix2D(eyes_center, angle, scale)
    M[0, 2] += (desired_face_width * 0.5 - eyes_center[0])
    M[1, 2] += (desired_face_width * 0.35 - eyes_center[1])
    return cv.warpAffine(image, M, (desired_face_width, desired_face_width), flags=cv.INTER_CUBIC)


def get_margin_crop(frame, bbox):
    h_img, w_img = frame.shape[:2]
    margin_x, margin_y = int(bbox.width * SPOOF_MARGIN_RATIO), int(bbox.height * SPOOF_MARGIN_RATIO)
    x1, y1 = max(0, bbox.origin_x - margin_x), max(0, bbox.origin_y - margin_y)
    x2, y2 = min(w_img, bbox.origin_x + bbox.width + margin_x), min(h_img, bbox.origin_y + bbox.height + margin_y)
    return frame[y1:y2, x1:x2]


def predict_spoof(crop_rgb):
    if crop_rgb is None or crop_rgb.size == 0:
        return None
    resized = cv.resize(crop_rgb, SPOOF_INPUT_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return float(spoof_model.predict_on_batch(batch)[0][0])


# --- MediaPipe Setup ---
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
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

with FaceDetector.create_from_options(options) as detector:
    cap = cv.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_timestamp = time.time()
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detector.detect_async(mp_image, int((current_timestamp - start_time) * 1000))

        if latest_result is not None:
            h, w = frame.shape[:2]
            active_face_keys = set()

            for idx, detection in enumerate(latest_result.detections):
                bbox = detection.bounding_box

                # 1. Identity Recognition
                identity_name = "Unknown"
                if len(detection.keypoints) >= 2:
                    eye_a = (detection.keypoints[0].x * w, detection.keypoints[0].y * h)
                    eye_b = (detection.keypoints[1].x * w, detection.keypoints[1].y * h)
                    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda p: p[0])
                    aligned = align_face(rgb_frame, left_eye, right_eye, FACENET_INPUT_SIZE)
                    face_emb = get_embedding(aligned)
                    identity_name = recognize_face(face_emb)

                # Generate a unique tracker key per-face (Name + Index to handle multiple Unknowns)
                face_key = f"{identity_name}_{idx}"
                active_face_keys.add(face_key)

                # Initialize tracking structures if new
                if face_key not in face_temporal_buffers:
                    face_temporal_buffers[face_key] = deque()
                    face_real_counters[face_key] = 0

                # 2. Get Single Frame Spoof Score
                spoof_crop = get_margin_crop(rgb_frame, bbox)
                spoof_score = predict_spoof(spoof_crop)

                display_label = "Processing..."
                text_color = (0, 255, 0)

                if spoof_score is not None:
                    is_frame_spoof = spoof_score < SPOOF_THRESHOLD

                    # --- ASYMMETRIC INSTANT RESET LOGIC ---
                    if not is_frame_spoof:
                        face_real_counters[face_key] += 1
                    else:
                        face_real_counters[face_key] = 0  # Break real streak instantly

                    # Instant Reset Condition
                    if face_real_counters[face_key] >= INSTANT_REAL_CONSECUTIVE_FRAMES:
                        face_temporal_buffers[face_key].clear()  # Erase all history instantly

                    # Append current frame to rolling temporal buffer
                    face_temporal_buffers[face_key].append((current_timestamp, is_frame_spoof))

                    # Clear out data points older than 3 seconds
                    while face_temporal_buffers[face_key] and (current_timestamp - face_temporal_buffers[face_key][0][0] > BUFFER_DURATION_SEC):
                        face_temporal_buffers[face_key].popleft()

                    # Calculate temporal percentage
                    total_samples = len(face_temporal_buffers[face_key])
                    spoof_samples = sum(1 for _, is_spf in face_temporal_buffers[face_key] if is_spf)
                    spoof_ratio = spoof_samples / total_samples if total_samples > 0 else 0.0

                    # Final State Decision
                    if spoof_ratio >= SPOOF_RATIO_TRIGGER:
                        display_label = f"SPOOFED: {identity_name}"
                        text_color = (0, 0, 255)
                    else:
                        display_label = identity_name
                        text_color = (0, 255, 0)

                    # Sub-metrics printing
                    cv.putText(frame, f"Ratio: {spoof_ratio:.2f} | Score: {spoof_score:.2f}",
                               (bbox.origin_x, bbox.origin_y + bbox.height + 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                # 3. Draw Bounding Boxes and UI
                cv.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                             (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), text_color, 2)
                cv.putText(frame, display_label, (bbox.origin_x, max(0, bbox.origin_y - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

                # 4. Draw Keypoints (right eye, left eye, nose, mouth, right ear, left ear)
                for kp_idx, keypoint in enumerate(detection.keypoints):
                    kp_x, kp_y = int(keypoint.x * w), int(keypoint.y * h)
                    if kp_idx < 2:
                        # The two eye points used for alignment - drawn larger and in yellow
                        cv.circle(frame, (kp_x, kp_y), 2, (0, 0, 255), -1)
                    else:
                        # Remaining landmarks (nose, mouth, ears) - drawn smaller and in cyan
                        cv.circle(frame, (kp_x, kp_y), 2, (0, 0, 255), -1)

            # Cleanup expired buffers for faces that left the camera frame view completely
            for dead_key in list(face_temporal_buffers.keys()):
                if dead_key not in active_face_keys:
                    del face_temporal_buffers[dead_key]
                    del face_real_counters[dead_key]

        # --- Interactive Keys ---
        cv.imshow('frame', frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Save the current face pointing directly to the center of frame
            if latest_result is not None and len(latest_result.detections) > 0:
                detection = latest_result.detections[0]  # Focus first item
                if len(detection.keypoints) >= 2:
                    eye_a = (detection.keypoints[0].x * w, detection.keypoints[0].y * h)
                    eye_b = (detection.keypoints[1].x * w, detection.keypoints[1].y * h)
                    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda p: p[0])
                    reg_face = align_face(rgb_frame, left_eye, right_eye, FACENET_INPUT_SIZE)
                    name = input("Enter name to Register: ").strip()
                    if name:
                        known_face_db[name] = get_embedding(reg_face)
                        save_database_to_json()

    cap.release()
    cv.destroyAllWindows()