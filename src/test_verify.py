# test_verify.py
import os
import json
import argparse
import cv2
import numpy as np


# ====== Signature calculation (same as in generator) ======
def _dct_phash(gray: np.ndarray, size=32, take=8) -> int:
    """Compute perceptual hash using DCT"""
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
    """Return Hamming distance between 64-bit integers"""
    return int(bin(a ^ b).count("1"))


def _hf_grid_strength(gray: np.ndarray, hf_step=4) -> float:
    h, w = gray.shape
    # BEFORE (bug):
    # pts = gray[0:h:hf_step, 0:w:w].astype(np.float32)
    # mask[0:h:hf_step, 0:w:w] = False

    # AFTER (fix):
    pts = gray[0:h:hf_step, 0:w:hf_step].astype(np.float32)
    mask = np.ones_like(gray, dtype=bool)
    mask[0:h:hf_step, 0:w:hf_step] = False

    grid_mean = float(pts.mean()) if pts.size else 0.0
    neigh = gray[mask].astype(np.float32)
    neigh_mean = float(neigh.mean()) if neigh.size else 0.0
    return grid_mean - neigh_mean


def _fft_peak_ratio(gray: np.ndarray, expect_cycles_x=4):
    """Ratio of expected frequency peak to average background"""
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
    """Return signature dictionary"""
    ph = _dct_phash(gray)
    hf = _hf_grid_strength(gray, hf_step=hf_step)
    fr = _fft_peak_ratio(gray, expect_cycles_x=lf_cycles_x)
    return {"phash": ph, "hf_strength": hf, "fft_peak_ratio": fr}


# ====== Print-scan degradation simulation ======
def simulate_print_scan(gray: np.ndarray) -> np.ndarray:
    """Simulate print -> scan -> reprint quality loss"""
    img = gray.copy()

    # Downscale then upscale (resolution loss)
    scale = np.random.uniform(0.6, 0.85)
    h, w = img.shape
    img_small = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    img = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Gaussian blur
    k = np.random.choice([3, 5])
    sigma = np.random.uniform(0.6, 1.4)
    img = cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

    # JPEG compression artifacts
    q = np.random.randint(35, 65)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    # Brightness/contrast adjustment
    alpha = np.random.uniform(0.9, 1.15)
    beta = np.random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gaussian noise
    noise = np.random.normal(0, 3.0, size=img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


# ====== Verification ======
def verify_with_signature(
    test_gray: np.ndarray,
    ref_sig: dict,
    hf_step=4,
    lf_cycles_x=4,
    phash_max_hamm=12,  # Max allowed Hamming distance
    hf_min_strength=0.4,  # Min HF grid strength
    fft_min_ratio=2.0,  # Min expected FFT peak ratio
):
    sig = compute_signature(test_gray, hf_step=hf_step, lf_cycles_x=lf_cycles_x)

    d_hamm = _hamming64(sig["phash"], int(ref_sig["phash"]))
    hf_ok = sig["hf_strength"] >= hf_min_strength
    fft_ok = sig["fft_peak_ratio"] >= fft_min_ratio
    ph_ok = d_hamm <= phash_max_hamm

    detail = {
        "hamming": d_hamm,
        "hf_strength": float(sig["hf_strength"]),
        "fft_peak_ratio": float(sig["fft_peak_ratio"]),
        "thresholds": {
            "phash_max_hamm": phash_max_hamm,
            "hf_min_strength": hf_min_strength,
            "fft_min_ratio": fft_min_ratio,
        },
    }
    passed = ph_ok and hf_ok and fft_ok
    return passed, detail


# ====== Generate fake copies for testing ======
def simulate_copies_for_folder(input_dir, out_dir, n_per_image=3):
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(input_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] cannot read {path}")
            continue
        for i in range(n_per_image):
            fake = simulate_print_scan(img)
            out = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_copy{i+1}.png")
            cv2.imwrite(out, fake)
            print(f"[FAKE] {out}")


# ====== Verify images in a folder ======
def verify_folder(test_dir, meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta["params"]
    items = {it["file"]: it["signature"] for it in meta["items"]}

    for name in os.listdir(test_dir):
        if not name.lower().endswith(".png"):
            continue
        p = os.path.join(test_dir, name)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[SKIP] cannot read {p}")
            continue

        ref_name = (
            name.replace("_copy1", "").replace("_copy2", "").replace("_copy3", "")
        )
        ref_sig = items.get(ref_name)
        if ref_sig is None:
            print(f"[WARN] no reference signature for {name} (expect {ref_name})")
            continue

        passed, detail = verify_with_signature(
            img, ref_sig, hf_step=params["hf_step"], lf_cycles_x=params["lf_cycles_x"]
        )
        verdict = "GENUINE" if passed else "COPY/ALTERED"
        print(f"{name}: {verdict}  |  detail={detail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["simulate", "verify"])
    ap.add_argument("--input_dir", required=True, help="Folder path")
    ap.add_argument(
        "--out_dir", default="simulated_copies", help="Output folder for simulate mode"
    )
    ap.add_argument(
        "--meta", default="signatures.json", help="Reference signatures JSON"
    )
    ap.add_argument(
        "--n", type=int, default=3, help="Number of copies per image (simulate mode)"
    )
    args = ap.parse_args()

    if args.mode == "simulate":
        simulate_copies_for_folder(args.input_dir, args.out_dir, n_per_image=args.n)
    else:
        verify_folder(args.input_dir, args.meta)
