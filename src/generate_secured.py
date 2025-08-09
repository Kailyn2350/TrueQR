# generate_secured.py
import os
import json
import argparse
import cv2
import numpy as np


# ---------- 1) Apply security patterns ----------
def add_security_patterns(
    img_gray: np.ndarray,
    brightness_variation=True,
    freq_pattern=True,
    phase_pattern=True,
    hf_step=4,  # spacing of high-frequency dots
    hf_delta=10.0,  # brightness change for high-frequency dots
    lf_cycles_x=4,  # sine wave cycles (low-frequency pattern)
):
    h, w = img_gray.shape
    secured = img_gray.copy().astype(np.float32)

    # 1) Pixel-wise Gaussian noise for brightness variation
    if brightness_variation:
        noise = np.random.normal(0, 5, size=(h, w)).astype(np.float32)
        secured = np.clip(secured + noise, 0, 255)

    # 2) Low + high-frequency patterns
    if freq_pattern:
        # High-frequency grid (dot pattern every hf_step pixels)
        hf = np.zeros((h, w), np.float32)
        for y in range(0, h, hf_step):
            for x in range(0, w, hf_step):
                hf[y, x] = hf_delta

        # Low-frequency horizontal sine wave
        xv, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        lf = (np.sin(xv * np.pi * lf_cycles_x) + 1.0) * 5.0

        secured = np.clip(secured + hf + lf, 0, 255)

    # 3) Slight phase modulation in Fourier domain
    if phase_pattern:
        dft = cv2.dft(secured, flags=cv2.DFT_COMPLEX_OUTPUT)
        mag, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])
        phase_shift = np.random.normal(0, 0.1, size=phase.shape).astype(np.float32)
        phase = phase + phase_shift
        real, imag = cv2.polarToCart(mag, phase)
        dft_mod = cv2.merge([real, imag])
        secured = cv2.idft(dft_mod, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        secured = np.clip(secured, 0, 255)

    return secured.astype(np.uint8)


# ---------- 2) Compute image signature ----------
def _dct_phash(gray: np.ndarray, size=32, take=8) -> int:
    """Perceptual hash (pHash) using DCT"""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    img = np.float32(img)
    dct = cv2.dct(img)
    dct_low = dct[:take, :take].copy()
    dct_low[0, 0] = 0.0
    med = np.median(dct_low)
    bits = (dct_low > med).astype(np.uint8).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return int(val)


def _hamming64(a: int, b: int) -> int:
    """Calculate Hamming distance between two 64-bit integers"""
    return int(bin(a ^ b).count("1"))


def _hf_grid_strength(gray: np.ndarray, hf_step=4) -> float:
    """Average difference between grid points and surrounding pixels"""
    h, w = gray.shape
    pts = gray[0:h:hf_step, 0:w:hf_step].astype(np.float32)
    grid_mean = float(pts.mean()) if pts.size else 0.0
    mask = np.ones_like(gray, dtype=bool)
    mask[0:h:hf_step, 0:w:hf_step] = False
    neigh = gray[mask].astype(np.float32)
    neigh_mean = float(neigh.mean()) if neigh.size else 0.0
    return grid_mean - neigh_mean


def _fft_peak_ratio(gray: np.ndarray, expect_cycles_x=4):
    """Peak energy ratio at expected frequency in horizontal spectrum"""
    h, w = gray.shape
    row = gray[h // 2, :].astype(np.float32)
    row = row - row.mean()
    spec = np.fft.rfft(row)
    mag = np.abs(spec)
    k = int(round(expect_cycles_x))
    k = max(1, min(k, len(mag) - 1))
    peak = mag[k]
    bg = (mag[1:]).mean() if len(mag) > 2 else 1.0
    return float(peak / (bg + 1e-6))


def compute_signature(gray: np.ndarray, hf_step=4, lf_cycles_x=4):
    """Return combined signature: pHash + HF strength + FFT peak ratio"""
    ph = _dct_phash(gray)
    hf = _hf_grid_strength(gray, hf_step=hf_step)
    fr = _fft_peak_ratio(gray, expect_cycles_x=lf_cycles_x)
    return {"phash": ph, "hf_strength": hf, "fft_peak_ratio": fr}


# ---------- 3) Batch process folder ----------
def process_folder(
    input_dir,
    output_dir,
    meta_path,
    brightness_variation=True,
    freq_pattern=True,
    phase_pattern=True,
    hf_step=4,
    hf_delta=10.0,
    lf_cycles_x=4,
):
    os.makedirs(output_dir, exist_ok=True)

    meta = {
        "params": {
            "brightness_variation": brightness_variation,
            "freq_pattern": freq_pattern,
            "phase_pattern": phase_pattern,
            "hf_step": hf_step,
            "hf_delta": hf_delta,
            "lf_cycles_x": lf_cycles_x,
        },
        "items": [],
    }

    for name in os.listdir(input_dir):
        if not name.lower().endswith(".png"):
            continue
        in_path = os.path.join(input_dir, name)
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] cannot read {in_path}")
            continue

        secured = add_security_patterns(
            img,
            brightness_variation=brightness_variation,
            freq_pattern=freq_pattern,
            phase_pattern=phase_pattern,
            hf_step=hf_step,
            hf_delta=hf_delta,
            lf_cycles_x=lf_cycles_x,
        )

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, secured)

        sig = compute_signature(secured, hf_step=hf_step, lf_cycles_x=lf_cycles_x)
        meta["items"].append({"file": name, "signature": sig})
        print(f"[OK] secured -> {out_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[META] signatures saved -> {meta_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with original PNGs")
    ap.add_argument("--output_dir", required=True, help="Folder to save secured PNGs")
    ap.add_argument(
        "--meta", default="signatures.json", help="Path to save signatures json"
    )
    args = ap.parse_args()

    process_folder(
        input_dir=args.input_dir, output_dir=args.output_dir, meta_path=args.meta
    )
