import base64
import io
import json
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, Response


# --- Verification Logic from test_verify.py ---


def _dct_phash(gray: np.ndarray, size=32, take=8) -> int:
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
    return int(bin(a ^ b).count("1"))


def _hf_grid_strength(gray: np.ndarray, hf_step=4) -> float:
    h, w = gray.shape
    pts = gray[0:h:hf_step, 0:w:hf_step].astype(np.float32)
    mask = np.ones_like(gray, dtype=bool)
    mask[0:h:hf_step, 0:w:hf_step] = False
    grid_mean = float(pts.mean()) if pts.size else 0.0
    neigh = gray[mask].astype(np.float32)
    neigh_mean = float(neigh.mean()) if neigh.size else 0.0
    return grid_mean - neigh_mean


def _fft_peak_ratio(gray: np.ndarray, expect_cycles_x=4):
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
    ph = _dct_phash(gray)
    hf = _hf_grid_strength(gray, hf_step=hf_step)
    fr = _fft_peak_ratio(gray, expect_cycles_x=lf_cycles_x)
    return {"phash": ph, "hf_strength": hf, "fft_peak_ratio": fr}


def verify_with_signature(
    test_gray: np.ndarray,
    ref_sig: dict,
    hf_step=4,
    lf_cycles_x=4,
    phash_max_hamm=18,
    hf_min_strength=0.15,
    fft_min_ratio=1.5,
):
    sig = compute_signature(test_gray, hf_step=hf_step, lf_cycles_x=lf_cycles_x)
    d_hamm = _hamming64(sig["phash"], int(ref_sig["phash"]))
    hf_ok = sig["hf_strength"] >= hf_min_strength
    fft_ok = sig["fft_peak_ratio"] >= fft_min_ratio
    ph_ok = d_hamm <= phash_max_hamm
    passed = ph_ok and hf_ok and fft_ok

    detail = {
        "passed_all": passed,
        "passed_phash": ph_ok,
        "passed_hf": hf_ok,
        "passed_fft": fft_ok,
        "hamming": d_hamm,
        "hf_strength": float(sig["hf_strength"]),
        "fft_peak_ratio": float(sig["fft_peak_ratio"]),
        "thresholds": {
            "phash_max_hamm": phash_max_hamm,
            "hf_min_strength": hf_min_strength,
            "fft_min_ratio": fft_min_ratio,
        },
    }
    return passed, detail


# --- Flask App ---

app = Flask(__name__, static_folder=".")
signatures_data = None
verification_params = None


def load_signatures():
    global signatures_data, verification_params
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        signatures_path = os.path.join(project_root, "signatures.json")

        with open(signatures_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        verification_params = meta["params"]
        signatures_data = meta["items"]
        print(f"[INFO] Loaded {len(signatures_data)} signatures from {signatures_path}")
    except Exception as e:
        print(f"[ERROR] Could not load signatures.json: {e}")
        signatures_data = []
        verification_params = {}


@app.before_request
def log_request_info():
    if request.path != "/favicon.ico":
        print(f"[REQUEST] Path: {request.path}, Method: {request.method}")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/verify", methods=["POST"])
def verify_image():
    if not signatures_data:
        return jsonify({"result": "Error: Signatures not loaded.", "detail": {}})

    try:
        data = request.get_json()
        image_data = data["image"]
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)
        image_np = np.frombuffer(binary_data, dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return (
                jsonify({"result": "Error: Image decoding failed.", "detail": {}}),
                400,
            )

        best_match = None
        closest_fail = None

        for ref in signatures_data:
            passed, detail = verify_with_signature(
                img,
                ref["signature"],
                hf_step=verification_params.get("hf_step", 4),
                lf_cycles_x=verification_params.get("lf_cycles_x", 4),
            )
            if passed:
                if (
                    best_match is None
                    or detail["hamming"] < best_match["detail"]["hamming"]
                ):
                    best_match = {
                        "result": "GENUINE ✅",
                        "detail": detail,
                        "ref_file": ref["file"],
                    }
            else:
                if (
                    closest_fail is None
                    or detail["hamming"] < closest_fail["detail"]["hamming"]
                ):
                    closest_fail = {
                        "result": "COPY / INVALID ❌",
                        "detail": detail,
                        "ref_file": ref["file"],
                    }

        if best_match:
            return jsonify(best_match)

        if closest_fail:
            return jsonify(closest_fail)

        return jsonify({"result": "No signatures to check against.", "detail": {}})

    except Exception as e:
        print(f"[ERROR] An exception occurred in /verify: {e}")
        return jsonify({"result": "Server Error", "detail": str(e)}), 500


@app.route("/<path:path>")
def serve_file(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    load_signatures()
    app.run(host="0.0.0.0", port=8000, debug=True)
