"""
enhance.py
Image quality improvement module. Both gates now run full multi-stage
pipelines that combine classical OpenCV operations with TensorFlow (ESRGAN)
as the final perceptual quality step.

  Gate 1 — "Severe Blur"  (5-stage pipeline)
  ─────────────────────────────────────────────
    1. Richardson-Lucy deconvolution (scipy)
         Iteratively inverts the blur convolution using an estimated PSF
         (motion or Gaussian, auto-detected from gradient orientation).
         Recovers structural edges that were mathematically present before
         the blur — something ESRGAN alone cannot do.
    2. Bilateral filter
         Suppresses ringing artefacts introduced by deconvolution while
         keeping recovered edges sharp.
    3. CLAHE on LAB L-channel (OpenCV)
         Restores local contrast that was compressed by the blur.
    4. ESRGAN  (TensorFlow Hub)
         4× generative super-resolution then resize back to original dims.
         Hallucinates realistic high-frequency texture on top of the
         structurally recovered image from steps 1-3.
    5. Unsharp mask
         Final sharpness lift to lock in the recovered detail.

  Gate 2 — "Accidental / Degraded Composition"  (6-stage pipeline)
  ──────────────────────────────────────────────────────────────────
    1. NLM denoising (OpenCV)
         Removes photon / read noise while preserving structural edges.
    2. CLAHE on LAB L-channel (OpenCV)
         Boosts local contrast without blowing highlights or crushing shadows.
    3. Auto white balance — grey-world assumption (numpy)
         Scales each BGR channel so its mean matches the global mean,
         correcting colour casts from poor lighting or ISO noise.
    4. ESRGAN  (TensorFlow Hub)
         Perceptual quality boost: recovers fine texture and sharpness lost
         to noise or poor exposure that classical filters cannot restore.
    5. Bilateral filter (OpenCV)
         Cleans up any compression / quantisation artefacts introduced by
         the ESRGAN resize step.
    6. Unsharp mask
         Final edge lift to compensate for the mild softening in step 1.
"""

import cv2
import numpy as np
import os

# ---------------------------------------------------------------------------
# Module-level ESRGAN model cache (shared by both Gate 1 and Gate 2 pipelines)
# ---------------------------------------------------------------------------
_esrgan_model = None


def _load_esrgan():
    global _esrgan_model
    if _esrgan_model is not None:
        return _esrgan_model
    try:
        import tensorflow_hub as hub
        print("[enhance] Loading ESRGAN fallback model from TensorFlow Hub...")
        _esrgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
        return _esrgan_model
    except Exception as exc:
        print(f"[enhance] ESRGAN unavailable: {exc}")
        return None


# ---------------------------------------------------------------------------
# PSF helpers
# ---------------------------------------------------------------------------

def _make_gaussian_psf_small(size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """Symmetric Gaussian PSF for defocus / radial blur."""
    ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return psf / psf.sum()


def _make_motion_psf(size: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """
    Linear motion-blur PSF of given length (size) and angle.
    angle_deg=0 → horizontal blur; 90 → vertical.
    """
    psf = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    for i in range(size):
        offset = int(round((i - center) * np.tan(np.radians(angle_deg))))
        col = center + offset
        if 0 <= col < size:
            psf[i, col] = 1.0
    total = psf.sum()
    return psf / total if total > 0 else _make_gaussian_psf_small(size)


def _estimate_psf(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    Analyses the image's gradient field to choose between a motion-blur
    PSF (strong dominant direction) and a Gaussian PSF (isotropic blur).

    A blurry image still retains some directional information in its
    gradient — motion blur produces gradients aligned perpendicular to the
    blur direction, while defocus blur produces isotropic gradients.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitudes and angles
    mag = np.sqrt(gx ** 2 + gy ** 2)
    strong = mag > np.percentile(mag, 75)   # top-25% edges only

    if strong.sum() == 0:
        return _make_gaussian_psf_small(kernel_size, sigma=3.0)

    angles = np.degrees(np.arctan2(gy[strong], gx[strong]))  # −180 … 180

    # Bin into 18 buckets of 20° each and find dominant direction
    hist, bin_edges = np.histogram(angles, bins=18, range=(-180, 180))
    dominant_bin = np.argmax(hist)
    dominant_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0
    dominance_ratio = hist[dominant_bin] / hist.sum()

    # If >35% of strong gradients share a direction, treat as motion blur
    if dominance_ratio > 0.35:
        # Motion blur direction is perpendicular to dominant gradient
        blur_angle = dominant_angle + 90.0
        return _make_motion_psf(kernel_size, blur_angle)

    return _make_gaussian_psf_small(kernel_size, sigma=3.0)


# ---------------------------------------------------------------------------
# Richardson-Lucy deconvolution
# ---------------------------------------------------------------------------

def _rl_deconvolve_channel(channel: np.ndarray,
                           psf: np.ndarray,
                           iterations: int = 25) -> np.ndarray:
    """
    Richardson-Lucy update rule for a single channel (float64, range 0-255):

        estimate_{n+1} = estimate_n * (PSF* ⊛ (channel / (PSF ⊛ estimate_n)))

    where ⊛ denotes convolution and PSF* is the spatially-flipped PSF.
    Implemented with FFT convolution for speed on large images.
    """
    from scipy.signal import fftconvolve

    img = channel.astype(np.float64)
    psf_f = psf.astype(np.float64)
    psf_mirror = np.flip(psf_f)

    estimate = img.copy()
    for _ in range(iterations):
        # Forward: convolve current estimate with PSF
        blurred_est = fftconvolve(estimate, psf_f, mode='same')
        blurred_est = np.maximum(blurred_est, 1e-8)   # avoid division by zero

        # Error ratio
        ratio = img / blurred_est

        # Backward: convolve ratio with flipped PSF (correlation step)
        correction = fftconvolve(ratio, psf_mirror, mode='same')

        # Update and clamp
        estimate = np.clip(estimate * correction, 0.0, 255.0)

    return estimate.astype(np.uint8)


def _richardson_lucy_enhance(image: np.ndarray,
                              iterations: int = 25) -> np.ndarray:
    """
    Full per-channel Richardson-Lucy deconvolution with automatic PSF
    estimation, followed by bilateral ringing suppression and a final
    mild unsharp mask.
    """
    psf = _estimate_psf(image, kernel_size=15)

    channels = []
    for c in range(3):
        ch = _rl_deconvolve_channel(image[:, :, c], psf, iterations)
        channels.append(ch)

    deblurred = np.stack(channels, axis=2)

    # Bilateral filter: suppresses ringing while preserving recovered edges
    result = cv2.bilateralFilter(deblurred, d=7, sigmaColor=25, sigmaSpace=25)
    return result


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------

def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE on the LAB L-channel — boosts local contrast without clipping."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _auto_white_balance(image: np.ndarray) -> np.ndarray:
    """Grey-world white balance: scales each BGR channel to match global mean."""
    result = image.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / (avg_b + 1e-8)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / (avg_g + 1e-8)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / (avg_r + 1e-8)), 0, 255)
    return result.astype(np.uint8)


def _esrgan_upscale(image: np.ndarray) -> np.ndarray:
    """
    ESRGAN 4× super-resolution then resize back to original dims.
    Falls back to unmodified image if TF is unavailable.
    """
    model = _load_esrgan()
    if model is None:
        return image
    import tensorflow as tf
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = tf.cast(tf.expand_dims(rgb, axis=0), tf.float32)
    out = model(tensor)
    enhanced = tf.cast(tf.clip_by_value(out[0], 0, 255), tf.uint8).numpy()
    restored = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)


def _unsharp_mask(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Unsharp mask: amplifies edges by (strength × original − (strength−1) × blur)."""
    blur = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image.astype(np.float32), strength, blur, -(strength - 1.0), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gate 1 enhancement: blur recovery
# ---------------------------------------------------------------------------

def enhance_blurry_image(image: np.ndarray) -> np.ndarray:
    """
    Gate 1 pipeline (5 stages):
      1. Richardson-Lucy deconvolution   (scipy; Wiener fallback if unavailable)
      2. Bilateral filter                (suppresses ringing — done inside R-L)
      3. CLAHE on LAB L-channel          (restores local contrast)
      4. ESRGAN 4× super-resolution      (hallucinates fine texture)
      5. Unsharp mask                    (final sharpness lift)
    """
    # Stage 1+2: deconvolve + bilateral ringing suppression
    try:
        from scipy.signal import fftconvolve
        _ = fftconvolve
        print("[enhance] Gate 1 — Stage 1: Richardson-Lucy deconvolution…")
        deblurred = _richardson_lucy_enhance(image, iterations=25)
    except ImportError:
        print("[enhance] scipy unavailable — using Wiener fallback for Stage 1.")
        deblurred = _wiener_fallback(image)

    # Stage 3: contrast restoration
    print("[enhance] Gate 1 — Stage 3: CLAHE…")
    contrasted = _apply_clahe(deblurred)

    # Stage 4: generative texture recovery
    print("[enhance] Gate 1 — Stage 4: ESRGAN…")
    upscaled = _esrgan_upscale(contrasted)

    # Stage 5: final sharpness lift
    print("[enhance] Gate 1 — Stage 5: unsharp mask…")
    return _unsharp_mask(upscaled, strength=1.5)



def _gaussian_psf_fft(h: int, w: int, sigma: float = 2.0) -> np.ndarray:
    """Gaussian PSF shifted to FFT origin convention (for Wiener fallback)."""
    ys = np.fft.fftfreq(h) * h
    xs = np.fft.fftfreq(w) * w
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    psf = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
    return psf / psf.sum()


def _wiener_fallback(image: np.ndarray) -> np.ndarray:
    """Last-resort Wiener deconvolution using numpy FFT only."""
    h, w = image.shape[:2]
    psf = _gaussian_psf_fft(h, w, sigma=2.0)
    H = np.fft.rfft2(psf)
    K = 0.005
    channels = []
    for c in range(3):
        ch = image[:, :, c].astype(np.float64) / 255.0
        W = np.conj(H) / (np.abs(H) ** 2 + K)
        deconv = np.fft.irfft2(np.fft.rfft2(ch) * W, s=(h, w))
        channels.append(np.clip(deconv * 255.0, 0, 255))
    deconvolved = np.stack(channels, axis=2).astype(np.uint8)
    result = cv2.bilateralFilter(deconvolved, d=5, sigmaColor=30, sigmaSpace=30)
    blur = cv2.GaussianBlur(result.astype(np.float32), (0, 0), sigmaX=1.0)
    return np.clip(cv2.addWeighted(result.astype(np.float32), 1.4, blur, -0.4, 0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gate 2 enhancement: noise / exposure / composition
# ---------------------------------------------------------------------------

def enhance_degraded_composition(image: np.ndarray) -> np.ndarray:
    """
    Gate 2 pipeline (6 stages):
      1. NLM denoising       (removes photon/read noise, preserves edges)
      2. CLAHE               (local contrast boost on LAB L-channel)
      3. Auto white balance  (grey-world correction for colour casts)
      4. ESRGAN              (generative texture + perceptual quality lift)
      5. Bilateral filter    (cleans compression artefacts from ESRGAN resize)
      6. Unsharp mask        (final edge lift to compensate NLM softening)
    """
    # Stage 1: denoise
    print("[enhance] Gate 2 — Stage 1: NLM denoising…")
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h=10, hColor=10,
        templateWindowSize=7, searchWindowSize=21
    )

    # Stage 2: local contrast
    print("[enhance] Gate 2 — Stage 2: CLAHE…")
    contrasted = _apply_clahe(denoised)

    # Stage 3: colour cast correction
    print("[enhance] Gate 2 — Stage 3: auto white balance…")
    balanced = _auto_white_balance(contrasted)

    # Stage 4: perceptual quality via ESRGAN
    print("[enhance] Gate 2 — Stage 4: ESRGAN…")
    upscaled = _esrgan_upscale(balanced)

    # Stage 5: suppress artefacts from ESRGAN resize
    print("[enhance] Gate 2 — Stage 5: bilateral filter…")
    cleaned = cv2.bilateralFilter(upscaled, d=5, sigmaColor=30, sigmaSpace=30)

    # Stage 6: final sharpness lift
    print("[enhance] Gate 2 — Stage 6: unsharp mask…")
    return _unsharp_mask(cleaned, strength=1.4)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def enhance_image(image_path: str, reason: str, output_dir: str = "enhanced_output") -> tuple:
    """
    Selects the correct enhancement strategy based on the degradation reason,
    saves the result, and returns its absolute path.

    Returns (True, enhanced_abs_path) on success, (False, "") on failure.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False, ""

    if "Blur" in reason:
        enhanced = enhance_blurry_image(image)
    else:
        enhanced = enhance_degraded_composition(image)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1] or ".jpg"
    enhanced_filename = f"{base_name}_enhanced{ext}"
    enhanced_path = os.path.join(output_dir, enhanced_filename)

    counter = 1
    while os.path.exists(enhanced_path):
        enhanced_filename = f"{base_name}_enhanced_{counter}{ext}"
        enhanced_path = os.path.join(output_dir, enhanced_filename)
        counter += 1

    if not cv2.imwrite(enhanced_path, enhanced):
        return False, ""

    return True, os.path.abspath(enhanced_path)
