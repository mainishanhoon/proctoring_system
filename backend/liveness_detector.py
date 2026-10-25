import random
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ── Landmark indices ───────────────────────────────────────

# Eye landmarks for blink detection
LEFT_EYE_TOP = [386, 387, 388, 390]
LEFT_EYE_BOTTOM = [374, 373, 372, 380]
LEFT_EYE_LEFT = [362]
LEFT_EYE_RIGHT = [263]

RIGHT_EYE_TOP = [159, 160, 161, 163]
RIGHT_EYE_BOTTOM = [145, 144, 143, 153]
RIGHT_EYE_LEFT = [33]
RIGHT_EYE_RIGHT = [133]

# Nose tip for head movement
NOSE_TIP = 1
NOSE_BRIDGE = 6
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Mouth landmarks for smile detection
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# ── Per-session liveness state ─────────────────────────────


class LivenessState:
    def __init__(self):
        self.blink_history = deque(maxlen=90)  # last 3 seconds at 30fps
        self.ear_history = deque(maxlen=30)
        self.head_pose_history = deque(maxlen=30)
        self.blink_count = 0
        self.last_blink_time = None
        self.frames_analysed = 0

        # Challenge state
        self.current_challenge = None
        self.challenge_start = None
        self.challenge_deadline = None
        self.challenge_result = None  # None, 'passed', 'failed'
        self.challenges_done = 0
        self.next_challenge_at = 30  # issue first challenge after 30 frames

    def to_dict(self):
        return {
            "blink_count": self.blink_count,
            "frames_analysed": self.frames_analysed,
            "challenges_done": self.challenges_done,
            "current_challenge": self.current_challenge,
            "challenge_result": self.challenge_result,
            "challenge_deadline": self.challenge_deadline,
        }


# Global state store — one per session_id
_session_states: dict = {}


def get_state(session_id: int) -> LivenessState:
    if session_id not in _session_states:
        _session_states[session_id] = LivenessState()
    return _session_states[session_id]


def clear_state(session_id: int):
    if session_id in _session_states:
        del _session_states[session_id]


# ── Eye Aspect Ratio (EAR) ─────────────────────────────────


def eye_aspect_ratio(landmarks, top_indices, bottom_indices, left_idx, right_idx, w, h):
    """
    EAR = vertical eye distance / horizontal eye distance
    When eye is open  → EAR ≈ 0.25-0.35
    When eye is closed → EAR ≈ 0.05-0.15
    """

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    top = np.mean([pt(i) for i in top_indices], axis=0)
    bottom = np.mean([pt(i) for i in bottom_indices], axis=0)
    left = pt(left_idx[0])
    right = pt(right_idx[0])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    return vertical / horizontal if horizontal > 0 else 0.0


# ── Head pose estimation ───────────────────────────────────


def estimate_head_pose(landmarks, w, h):
    """
    Returns yaw (left/right) and pitch (up/down) estimates
    based on nose and cheek positions.
    """

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    nose = pt(NOSE_TIP)
    left_cheek = pt(LEFT_CHEEK)
    right_cheek = pt(RIGHT_CHEEK)
    chin = pt(CHIN)
    bridge = pt(NOSE_BRIDGE)

    # Yaw — how much nose deviates from cheek midpoint horizontally
    cheek_mid_x = (left_cheek[0] + right_cheek[0]) / 2
    face_width = abs(right_cheek[0] - left_cheek[0])
    yaw = (nose[0] - cheek_mid_x) / face_width if face_width > 0 else 0

    # Pitch — nose position relative to chin and bridge
    face_height = abs(chin[1] - bridge[1])
    pitch = (nose[1] - bridge[1]) / face_height if face_height > 0 else 0.5

    return {"yaw": round(yaw, 3), "pitch": round(pitch, 3)}


# ── Smile detection ────────────────────────────────────────


def detect_smile(landmarks, w, h):
    """
    Estimates smile by mouth width vs mouth height ratio.
    Smiling → wider mouth, small vertical opening.
    """

    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    left = pt(MOUTH_LEFT)
    right = pt(MOUTH_RIGHT)
    top = pt(MOUTH_TOP)
    bottom = pt(MOUTH_BOTTOM)

    width = np.linalg.norm(right - left)
    height = np.linalg.norm(bottom - top)

    ratio = width / height if height > 0 else 0
    # Typical smile ratio > 4.0
    return ratio > 4.0, round(ratio, 2)


# ── Challenge management ───────────────────────────────────

CHALLENGES = [
    {"id": "blink", "instruction": "Please blink twice", "timeout": 8},
    {
        "id": "turn_left",
        "instruction": "Please slowly turn your head to the LEFT",
        "timeout": 8,
    },
    {
        "id": "turn_right",
        "instruction": "Please slowly turn your head to the RIGHT",
        "timeout": 8,
    },
    {"id": "smile", "instruction": "Please smile", "timeout": 6},
    {
        "id": "nod",
        "instruction": "Please nod your head (move up and down)",
        "timeout": 8,
    },
]


def issue_challenge(state: LivenessState) -> dict:
    challenge = random.choice(CHALLENGES)
    state.current_challenge = challenge["id"]
    state.challenge_start = datetime.utcnow()
    state.challenge_deadline = challenge["timeout"]
    state.challenge_result = None
    return {
        "type": "challenge",
        "challenge_id": challenge["id"],
        "instruction": challenge["instruction"],
        "timeout": challenge["timeout"],
    }


def check_challenge_completion(
    state: LivenessState, blinked: bool, pose: dict, smiled: bool, prev_pitch: float
) -> str:
    """
    Returns 'passed', 'failed', or 'pending'
    """
    if not state.current_challenge:
        return "none"

    elapsed = (datetime.utcnow() - state.challenge_start).total_seconds()

    # Timeout check
    if elapsed > state.challenge_deadline:
        return "failed"

    c = state.current_challenge

    if c == "blink" and blinked:
        return "passed"

    if c == "turn_left" and pose["yaw"] < -0.25:
        return "passed"

    if c == "turn_right" and pose["yaw"] > 0.25:
        return "passed"

    if c == "smile" and smiled:
        return "passed"

    if c == "nod":
        pitch_change = abs(pose["pitch"] - prev_pitch) if prev_pitch else 0
        if pitch_change > 0.15:
            return "passed"

    return "pending"


# ── Main liveness analysis ─────────────────────────────────

EAR_BLINK_THRESHOLD = 0.18
MIN_BLINKS_PER_MINUTE = 4


def analyse_liveness(image_bytes: bytes, session_id: int) -> dict:
    """
    Runs passive liveness analysis on one frame.
    Returns liveness verdict and any challenge instructions.
    """
    state = get_state(session_id)
    state.frames_analysed += 1

    result = {
        "is_live": True,
        "liveness_score": 1.0,
        "violations": [],
        "challenge": None,
        "challenge_status": state.challenge_result,
        "blink_count": state.blink_count,
        "state": state.to_dict(),
    }

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        mesh_result = face_mesh.process(rgb)

        if not mesh_result.multi_face_landmarks:
            # No face — can't assess liveness
            result["is_live"] = False
            result["liveness_score"] = 0.0
            return result

        landmarks = mesh_result.multi_face_landmarks[0].landmark

        # ── EAR calculation ────────────────────────────────
        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE_TOP,
            LEFT_EYE_BOTTOM,
            LEFT_EYE_LEFT,
            LEFT_EYE_RIGHT,
            w,
            h,
        )
        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE_TOP,
            RIGHT_EYE_BOTTOM,
            RIGHT_EYE_LEFT,
            RIGHT_EYE_RIGHT,
            w,
            h,
        )
        avg_ear = (left_ear + right_ear) / 2
        state.ear_history.append(avg_ear)

        # Blink detection — EAR drops below threshold
        blinked = False
        if len(state.ear_history) >= 3:
            if (
                state.ear_history[-2] < EAR_BLINK_THRESHOLD
                and state.ear_history[-1] >= EAR_BLINK_THRESHOLD
                and state.ear_history[-3] >= EAR_BLINK_THRESHOLD
            ):
                blinked = True
                state.blink_count += 1
                state.last_blink_time = datetime.utcnow()

        result["blink_count"] = state.blink_count

        # ── Head pose ──────────────────────────────────────
        pose = estimate_head_pose(landmarks, w, h)
        prev_pitch = (
            state.head_pose_history[-1]["pitch"] if state.head_pose_history else None
        )
        state.head_pose_history.append(pose)

        # ── Smile detection ────────────────────────────────
        smiled, smile_ratio = detect_smile(landmarks, w, h)

        # ── No-blink check (passive spoofing signal) ───────
        # If 150 frames (~5 seconds) and zero blinks → suspicious
        if (
            state.frames_analysed % 150 == 0
            and state.frames_analysed > 0
            and state.blink_count == 0
        ):
            result["is_live"] = False
            result["liveness_score"] = 0.2
            result["violations"].append(
                {
                    "type": "no_blink_detected",
                    "severity": "high",
                    "description": "No blinking detected — possible photo spoof",
                }
            )

        # ── Blink rate check ───────────────────────────────
        # After 600 frames (~20 seconds), check blink rate
        if state.frames_analysed == 600:
            blinks_per_minute = state.blink_count * 3
            if blinks_per_minute < MIN_BLINKS_PER_MINUTE:
                result["violations"].append(
                    {
                        "type": "low_blink_rate",
                        "severity": "medium",
                        "description": f"Unusually low blink rate: {blinks_per_minute:.0f} bpm",
                    }
                )
                result["liveness_score"] = max(0.4, result["liveness_score"] - 0.3)

        # ── No head movement check ─────────────────────────
        if len(state.head_pose_history) == 30 and state.frames_analysed % 90 == 0:
            yaw_values = [p["yaw"] for p in state.head_pose_history]
            yaw_range = max(yaw_values) - min(yaw_values)
            if yaw_range < 0.02:
                result["violations"].append(
                    {
                        "type": "no_head_movement",
                        "severity": "medium",
                        "description": "No head movement detected — possible static image",
                    }
                )
                result["liveness_score"] = max(0.4, result["liveness_score"] - 0.2)

        # ── Challenge management ───────────────────────────
        if state.current_challenge:
            challenge_outcome = check_challenge_completion(
                state, blinked, pose, smiled, prev_pitch
            )

            if challenge_outcome == "passed":
                state.challenge_result = "passed"
                state.current_challenge = None
                state.challenges_done += 1
                result["challenge_status"] = "passed"

            elif challenge_outcome == "failed":
                state.challenge_result = "failed"
                state.current_challenge = None
                result["challenge_status"] = "failed"
                result["violations"].append(
                    {
                        "type": "challenge_failed",
                        "severity": "high",
                        "description": "Student failed liveness challenge — possible spoofing",
                    }
                )
                result["is_live"] = False
                result["liveness_score"] = 0.3

            else:
                result["challenge_status"] = "pending"

        # Issue a new challenge when due
        elif state.frames_analysed >= state.next_challenge_at:
            challenge_msg = issue_challenge(state)
            result["challenge"] = challenge_msg
            state.next_challenge_at = state.frames_analysed + random.randint(60, 120)

        result["state"] = state.to_dict()

    except Exception as e:
        print(f"Liveness analysis error: {e}")

    return result
