import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from liveness_detector import analyse_liveness
from deepfake_detector import analyse_deepfake
from face_verifier import verify_face

# Initialise MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_face_mesh     = mp.solutions.face_mesh
mp_drawing       = mp.solutions.drawing_utils

# Load models once — not on every frame
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,      # 0 = short range (webcam), 1 = long range
    min_detection_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,        # detect up to 5 faces
    refine_landmarks=True,  # enables iris tracking
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# MediaPipe iris landmark indices
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corner indices for gaze ratio calculation
LEFT_EYE_CORNERS  = [33, 133]   # inner, outer corners
RIGHT_EYE_CORNERS = [362, 263]

# ── Main detection function ───────────────────────────────

def analyse_frame(image_bytes: bytes,
                  session_id: int = 0,
                  stored_encoding: str = None) -> dict:
    """
    Runs face detection, liveness analysis, and deepfake detection
    on a single webcam frame. Returns combined result.
    """
    result = {
        "face_count":       0,
        "face_detected":    False,
        "multiple_faces":   False,
        "gaze_direction":   "center",
        "gaze_suspicious":  False,
        "œ":         None,
        "deepfake":         None,
        "violations":       [],
        "timestamp":        datetime.utcnow().isoformat()
    }

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            result["violations"].append({
                "type":        "invalid_frame",
                "severity":    "low",
                "description": "Could not decode image frame"
            })
            return result

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # ── Face detection ─────────────────────────────────
        face_results = face_detector.process(rgb_frame)

        if face_results.detections:
            result["face_count"]    = len(face_results.detections)
            result["face_detected"] = True

            if result["face_count"] > 1:
                result["multiple_faces"] = True
                result["violations"].append({
                    "type":        "multiple_faces",
                    "severity":    "high",
                    "description": f"{result['face_count']} faces detected in frame"
                })
        else:
            result["violations"].append({
                "type":        "no_face",
                "severity":    "medium",
                "description": "No face detected — student may have left"
            })
            return result

        # ── Gaze detection ─────────────────────────────────
        mesh_results = face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            gaze      = calculate_gaze(landmarks, w, h)
            result["gaze_direction"] = gaze["direction"]
            result["gaze_suspicious"] = gaze["suspicious"]

            if gaze["suspicious"]:
                result["violations"].append({
                    "type":        "suspicious_gaze",
                    "severity":    "medium",
                    "description": f"Gaze direction: {gaze['direction']}"
                })

        # ── Liveness detection ─────────────────────────────
        liveness = analyse_liveness(image_bytes, session_id)
        result["liveness"] = liveness

        for v in liveness.get("violations", []):
            result["violations"].append(v)

        # ── Deepfake detection ─────────────────────────────
        deepfake = analyse_deepfake(image_bytes)
        result["deepfake"] = deepfake

        if deepfake["verdict"] in ("likely_deepfake", "suspicious"):
            result["violations"].append({
                "type":        "deepfake_detected",
                "severity":    "high",
                "description": f"Deepfake/spoofing detected — verdict: {deepfake['verdict']}, score: {deepfake['deepfake_score']}"
            })
        # ── Face verification ──────────────────────────────
        if stored_encoding:
            verification = verify_face(image_bytes, stored_encoding)
            result["verification"] = verification

            if verification["violation"]:
                result["violations"].append(verification["violation"])
        
    except Exception as e:
        print(f"Frame analysis error: {e}")
        result["violations"].append({
            "type":        "detection_error",
            "severity":    "low",
            "description": f"Detection error: {str(e)}"
        })

    return result

# ── Gaze calculation ──────────────────────────────────────

def calculate_gaze(landmarks, frame_width: int, frame_height: int) -> dict:
    """
    Uses iris position relative to eye corners to
    estimate where the student is looking.
    """
    try:
        def get_point(idx):
            lm = landmarks[idx]
            return int(lm.x * frame_width), int(lm.y * frame_height)

        # Get iris centres
        left_iris_points  = [get_point(i) for i in LEFT_IRIS]
        right_iris_points = [get_point(i) for i in RIGHT_IRIS]

        left_iris_centre  = np.mean(left_iris_points,  axis=0)
        right_iris_centre = np.mean(right_iris_points, axis=0)

        # Get eye corners
        left_inner  = get_point(LEFT_EYE_CORNERS[0])
        left_outer  = get_point(LEFT_EYE_CORNERS[1])
        right_inner = get_point(RIGHT_EYE_CORNERS[0])
        right_outer = get_point(RIGHT_EYE_CORNERS[1])

        # Calculate horizontal gaze ratio for each eye
        # ratio < 0.35 = looking left, > 0.65 = looking right
        def horizontal_ratio(iris_centre, inner, outer):
            eye_width = abs(outer[0] - inner[0])
            if eye_width == 0:
                return 0.5
            iris_offset = iris_centre[0] - min(inner[0], outer[0])
            return iris_offset / eye_width

        left_ratio  = horizontal_ratio(left_iris_centre,  left_inner,  left_outer)
        right_ratio = horizontal_ratio(right_iris_centre, right_inner, right_outer)
        avg_ratio   = (left_ratio + right_ratio) / 2

        # Calculate vertical gaze using nose tip vs forehead
        nose_tip  = get_point(1)
        forehead  = get_point(10)
        chin      = get_point(152)

        face_height   = abs(chin[1] - forehead[1])
        nose_position = (nose_tip[1] - forehead[1]) / face_height if face_height > 0 else 0.5

        # Determine direction
        if avg_ratio < 0.35:
            direction = "left"
            suspicious = True
        elif avg_ratio > 0.65:
            direction = "right"
            suspicious = True
        elif nose_position < 0.35:
            direction = "up"
            suspicious = True
        elif nose_position > 0.70:
            direction = "down"
            suspicious = True
        else:
            direction = "center"
            suspicious = False

        return {
            "direction":  direction,
            "suspicious": suspicious,
            "ratio":      round(avg_ratio, 3)
        }

    except Exception:
        return {"direction": "unknown", "suspicious": False, "ratio": 0.5}