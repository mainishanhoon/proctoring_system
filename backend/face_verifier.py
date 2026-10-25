import json

import cv2
import numpy as np

# Cosine similarity threshold for embedding comparison
MATCH_THRESHOLD = 0.70

# OpenCV face detector — already installed, no download
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# LBPH face recogniser — built into OpenCV
recogniser = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)


def detect_and_crop_face(frame: np.ndarray, target_size=(128, 128)):
    """
    Detects the largest face in frame and returns
    a normalised grayscale crop.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, None

    # Largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding
    pad = int(0.15 * min(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    face_crop = gray[y1:y2, x1:x2]

    # Equalise histogram for lighting normalisation
    face_crop = cv2.equalizeHist(face_crop)

    # Resize to fixed size
    face_crop = cv2.resize(face_crop, target_size)

    return face_crop, (x1, y1, x2, y2)


def extract_encoding(image_bytes: bytes) -> list | None:
    """
    Extracts a face descriptor from image bytes.
    Returns a flat list representing the face for storage.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        face_crop, bbox = detect_and_crop_face(frame)

        if face_crop is None:
            return None

        # Use HOG descriptor for face embedding
        # More robust than raw pixels for lighting variation
        win_size = (128, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        descriptor = hog.compute(face_crop)

        if descriptor is None:
            return None

        # L2 normalise
        descriptor = descriptor.flatten()
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        return descriptor.tolist()

    except Exception as e:
        print(f"Encoding extraction error: {e}")
        return None


def encoding_to_str(encoding: list) -> str:
    """Serialise encoding to string for database storage"""
    return json.dumps(encoding)


def str_to_encoding(encoding_str: str) -> np.ndarray:
    """Deserialise encoding from database"""
    return np.array(json.loads(encoding_str))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def verify_face(image_bytes: bytes, stored_encoding_str: str) -> dict:
    """
    Compares face in current frame to registered encoding.
    """
    result = {
        "verified": False,
        "distance": 1.0,
        "confidence": 0.0,
        "status": "unknown",
        "violation": None,
    }

    try:
        stored_np = str_to_encoding(stored_encoding_str)
        current_encoding = extract_encoding(image_bytes)

        if current_encoding is None:
            result["status"] = "no_face"
            return result

        current_np = np.array(current_encoding)
        similarity = cosine_similarity(stored_np, current_np)
        distance = 1.0 - similarity

        result["distance"] = round(distance, 4)
        result["confidence"] = round(similarity, 4)

        if similarity >= MATCH_THRESHOLD:
            result["verified"] = True
            result["status"] = "match"
            print(f"Face verified — similarity: {similarity:.1%}")
        else:
            result["verified"] = False
            result["status"] = "mismatch"
            print(f"Face mismatch — similarity: {similarity:.1%}")
            result["violation"] = {
                "type": "wrong_person",
                "severity": "high",
                "description": (
                    f"Face does not match registered student. "
                    f"Similarity: {similarity:.1%} "
                    f"(required: {MATCH_THRESHOLD:.1%})"
                ),
            }

    except Exception as e:
        print(f"Face verification error: {e}")
        result["status"] = "error"

    return result
