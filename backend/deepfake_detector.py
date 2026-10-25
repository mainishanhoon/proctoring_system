import cv2
import mediapipe as mp
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6
)

# ── Deepfake detection via frequency analysis ──────────────


def analyse_deepfake(image_bytes: bytes) -> dict:
    """
    Analyses a frame for deepfake/GAN artifacts using:
    1. FFT frequency domain analysis — GANs leave periodic artifacts
    2. Noise pattern analysis — real cameras have natural sensor noise
    3. Facial region inconsistency — face-swaps often have blending edges
    """
    result = {
        "is_authentic": True,
        "deepfake_score": 0.0,  # 0 = real, 1 = likely fake
        "flags": [],
        "verdict": "authentic",
    }

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # ── Step 1: Extract face region ────────────────────
        face_region = extract_face_region(rgb, gray, w, h)
        if face_region is None:
            # No face found — skip deepfake analysis
            return result

        face_gray, face_color, face_bbox = face_region

        # ── Step 2: FFT frequency analysis ────────────────
        fft_score, fft_flags = fft_analysis(face_gray)
        result["flags"].extend(fft_flags)

        # ── Step 3: Noise pattern analysis ────────────────
        noise_score, noise_flags = noise_analysis(face_gray)
        result["flags"].extend(noise_flags)

        # ── Step 4: Blending boundary check ───────────────
        blend_score, blend_flags = blending_analysis(frame, face_bbox, w, h)
        result["flags"].extend(blend_flags)

        # ── Combined score ─────────────────────────────────
        # Weighted combination of all three analyses
        combined = (fft_score * 0.40) + (noise_score * 0.35) + (blend_score * 0.25)

        result["deepfake_score"] = round(combined, 3)

        # Verdict thresholds
        if combined > 0.65:
            result["is_authentic"] = False
            result["verdict"] = "likely_deepfake"
        elif combined > 0.45:
            result["is_authentic"] = False
            result["verdict"] = "suspicious"
        else:
            result["verdict"] = "authentic"

    except Exception as e:
        print(f"Deepfake analysis error: {e}")

    return result


def extract_face_region(rgb, gray, w, h):
    """Extract just the face crop for focused analysis"""
    detection_result = face_detector.process(rgb)

    if not detection_result.detections:
        return None

    detection = detection_result.detections[0]
    bbox = detection.location_data.relative_bounding_box

    # Convert relative coords to pixel coords
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width) * w))
    y2 = min(h, int((bbox.ymin + bbox.height) * h))

    if x2 - x1 < 40 or y2 - y1 < 40:
        return None  # Face too small to analyse

    face_gray = gray[y1:y2, x1:x2]
    face_color = rgb[y1:y2, x1:x2]

    # Resize for consistent analysis
    face_gray = cv2.resize(face_gray, (128, 128))
    face_color = cv2.resize(face_color, (128, 128))

    return face_gray, face_color, (x1, y1, x2, y2)


def fft_analysis(face_gray: np.ndarray) -> tuple:
    """
    GAN-generated faces often show periodic patterns in the
    frequency domain due to upsampling artifacts.
    Real faces show smooth, organic frequency distributions.
    """
    flags = []
    score = 0.0

    try:
        # Compute FFT
        f_transform = fft2(face_gray.astype(float))
        f_shifted = fftshift(f_transform)
        magnitude = np.log(np.abs(f_shifted) + 1)

        # Normalise
        magnitude = (magnitude - magnitude.min()) / (
            magnitude.max() - magnitude.min() + 1e-8
        )

        # Check for periodic peaks — GAN signature
        # Real images: smooth falloff from centre
        # GAN images: anomalous peaks at regular intervals
        centre = magnitude.shape[0] // 2
        quadrant = magnitude[centre - 32 : centre + 32, centre - 32 : centre + 32]
        mean_q = quadrant.mean()
        std_q = quadrant.std()

        # High-frequency energy ratio
        hf_region = np.concatenate(
            [
                magnitude[:20, :].flatten(),
                magnitude[-20:, :].flatten(),
                magnitude[:, :20].flatten(),
                magnitude[:, -20:].flatten(),
            ]
        )
        hf_energy = hf_region.mean()

        # GAN images tend to have unusual high-frequency energy
        if hf_energy > 0.45:
            score += 0.4
            flags.append("unusual_high_frequency_energy")

        # Check for periodic grid pattern (checkerboard artifact)
        row_var = np.var(magnitude, axis=1)
        col_var = np.var(magnitude, axis=0)

        if np.max(row_var) / (np.mean(row_var) + 1e-8) > 5.0:
            score += 0.3
            flags.append("periodic_row_artifact")

        if np.max(col_var) / (np.mean(col_var) + 1e-8) > 5.0:
            score += 0.3
            flags.append("periodic_column_artifact")

    except Exception as e:
        print(f"FFT analysis error: {e}")

    return min(score, 1.0), flags


def noise_analysis(face_gray: np.ndarray) -> tuple:
    """
    Real camera images have natural sensor noise with specific
    statistical properties. GAN images have different noise
    characteristics — often too smooth or with synthetic patterns.
    """
    flags = []
    score = 0.0

    try:
        face_float = face_gray.astype(float)

        # Estimate noise using Laplacian
        laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
        noise_std = laplacian.std()

        # Real webcam images: moderate noise (std typically 15-60)
        # Photos held to camera: low noise
        # GAN faces: very low or unusual noise patterns
        if noise_std < 8.0:
            score += 0.4
            flags.append("unnaturally_low_noise")
        elif noise_std > 120.0:
            score += 0.2
            flags.append("unusually_high_noise")

        # Check local variance uniformity
        # Real faces: varying texture (skin, eyes, hair)
        # Smoothed GAN faces: unnaturally uniform regions
        kernel_size = 8
        local_means = ndimage.uniform_filter(face_float, size=kernel_size)
        local_vars = (
            ndimage.uniform_filter(face_float**2, size=kernel_size) - local_means**2
        )
        local_vars = np.maximum(local_vars, 0)

        var_of_vars = np.var(local_vars)

        if var_of_vars < 500:
            score += 0.35
            flags.append("uniform_texture_distribution")

        # JPEG compression artifact check
        # Real webcam → natural artifacts
        # Screen recordings / virtual cameras → different artifact patterns
        dct_check = dct_block_analysis(face_gray)
        if dct_check > 0.6:
            score += 0.25
            flags.append("unusual_compression_pattern")

    except Exception as e:
        print(f"Noise analysis error: {e}")

    return min(score, 1.0), flags


def dct_block_analysis(gray: np.ndarray) -> float:
    """
    Analyses 8x8 DCT blocks for compression artifact patterns.
    Virtual cameras and screen captures show different blocking
    characteristics than real camera footage.
    """
    try:
        h, w = gray.shape
        block_size = 8
        scores = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y : y + block_size, x : x + block_size].astype(float)
                block_dct = cv2.dct(block)
                # Energy in high-frequency DCT coefficients
                hf_energy = np.sum(np.abs(block_dct[4:, 4:]))
                scores.append(hf_energy)

        if not scores:
            return 0.0

        scores = np.array(scores)
        mean_hf = np.mean(scores)
        std_hf = np.std(scores)
        cv = std_hf / (mean_hf + 1e-8)  # Coefficient of variation

        # Unusually low or high variation in block energies is suspicious
        if cv < 0.3 or cv > 3.0:
            return 0.7
        return 0.1

    except Exception:
        return 0.0


def blending_analysis(frame: np.ndarray, face_bbox: tuple, w: int, h: int) -> tuple:
    """
    Face-swap deepfakes often have visible blending artifacts at
    the boundary between the swapped face and the real neck/hair.
    We check for colour and gradient discontinuities at the face edge.
    """
    flags = []
    score = 0.0

    try:
        x1, y1, x2, y2 = face_bbox

        # Expand bbox slightly to capture boundary region
        margin = 10
        bx1 = max(0, x1 - margin)
        by1 = max(0, y1 - margin)
        bx2 = min(w, x2 + margin)
        by2 = min(h, y2 + margin)

        boundary_region = frame[by1:by2, bx1:bx2]

        if boundary_region.size == 0:
            return score, flags

        # Compute gradient magnitude
        gray_boundary = cv2.cvtColor(boundary_region, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_boundary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_boundary, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Look at just the border strip
        border = 12
        border_grad = np.concatenate(
            [
                gradient_mag[:border, :].flatten(),
                gradient_mag[-border:, :].flatten(),
                gradient_mag[:, :border].flatten(),
                gradient_mag[:, -border:].flatten(),
            ]
        )

        inner_grad = gradient_mag[border:-border, border:-border].flatten()

        if len(border_grad) > 0 and len(inner_grad) > 0:
            border_mean = np.mean(border_grad)
            inner_mean = np.mean(inner_grad) + 1e-8
            ratio = border_mean / inner_mean

            # Strong gradients at face boundary = blending artifact
            if ratio > 2.5:
                score += 0.5
                flags.append("boundary_blending_artifact")

        # Colour consistency check between face and surrounding skin
        if x2 - x1 > 60 and y2 - y1 > 60:
            # Sample colours at face edge and just outside
            face_edge = frame[y1 : y1 + 15, x1:x2]
            outside_top = frame[max(0, y1 - 15) : y1, x1:x2]

            if outside_top.size > 0:
                face_edge_mean = face_edge.mean(axis=(0, 1))
                outside_edge_mean = outside_top.mean(axis=(0, 1))
                colour_diff = np.abs(face_edge_mean - outside_edge_mean).mean()

                # Large colour jump at boundary → face swap artifact
                if colour_diff > 35:
                    score += 0.4
                    flags.append("colour_discontinuity_at_boundary")

    except Exception as e:
        print(f"Blending analysis error: {e}")

    return min(score, 1.0), flags
