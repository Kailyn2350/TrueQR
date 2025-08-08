const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const banner = document.getElementById("banner");
const debugInfo = document.createElement("div");
document.body.appendChild(debugInfo);

const ctx = canvas.getContext("2d");

let isRequesting = false;

debugInfo.style.position = "absolute";
debugInfo.style.bottom = "100px";
debugInfo.style.left = "50%";
debugInfo.style.transform = "translateX(-50%)";
debugInfo.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
debugInfo.style.padding = "10px";
debugInfo.style.borderRadius = "5px";
debugInfo.style.fontSize = "0.9em";
debugInfo.style.whiteSpace = "pre";
debugInfo.style.fontFamily = "monospace";
debugInfo.style.display = "none";

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });
    video.srcObject = stream;
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });
  } catch (err) {
    console.error("Camera Error:", err);
    banner.textContent = "Could not access the camera.";
    banner.style.backgroundColor = "red";
  }
}

function drawMask() {
  const w = canvas.width;
  const h = canvas.height;
  const boxSize = Math.min(w, h) * 0.7;
  const left = (w - boxSize) / 2;
  const top = (h - boxSize) / 2;

  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(0, 0, w, h);
  ctx.globalCompositeOperation = "destination-out";
  ctx.fillRect(left, top, boxSize, boxSize);
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 3;
  ctx.strokeRect(left, top, boxSize, boxSize);
  ctx.restore();
}

async function verifyFrame() {
  if (isRequesting || !video.srcObject) return;
  isRequesting = true;

  const w = video.videoWidth;
  const h = video.videoHeight;
  const boxSize = Math.min(w, h) * 0.7;
  const sx = (w - boxSize) / 2;
  const sy = (h - boxSize) / 2;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = boxSize;
  tempCanvas.height = boxSize;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(video, sx, sy, boxSize, boxSize, 0, 0, boxSize, boxSize);
  
  // --- Changed to PNG for lossless compression ---
  const imageDataUrl = tempCanvas.toDataURL("image/png");

  try {
    const response = await fetch("/verify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageDataUrl }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    updateBanner(data.result, data.detail);

  } catch (error) {
    console.error("Verification failed:", error);
    updateBanner("Verification Error", {});
  } finally {
    setTimeout(() => { isRequesting = false; }, 500);
  }
}

function updateBanner(result, detail) {
  banner.textContent = result;
  if (result.includes("GENUINE")) {
    banner.className = "detect-banner genuine";
    debugInfo.style.display = "none";
  } else if (result.includes("COPY")) {
    banner.className = "detect-banner fake";
    displayDebugInfo(detail);
  } else {
    banner.className = "detect-banner";
    debugInfo.style.display = "none";
  }
}

function displayDebugInfo(detail) {
  if (!detail || Object.keys(detail).length === 0) {
    debugInfo.style.display = "none";
    return;
  }
  
  const { hamming, hf_strength, fft_peak_ratio, thresholds } = detail;
  const th = thresholds || {};

  const hf_str_fixed = hf_strength ? hf_strength.toFixed(3) : "N/A";
  const fft_rat_fixed = fft_peak_ratio ? fft_peak_ratio.toFixed(3) : "N/A";

  debugInfo.textContent = 
`--[Debug Info]--
` + 
`pHash Dist: ${hamming} (Max: ${th.phash_max_hamm}) ${detail.passed_phash ? '✅' : '❌'}
` + 
`HF Strength: ${hf_str_fixed} (Min: ${th.hf_min_strength}) ${detail.passed_hf ? '✅' : '❌'}
` + 
`FFT Ratio: ${fft_rat_fixed} (Min: ${th.fft_min_ratio}) ${detail.passed_fft ? '✅' : '❌'}`;

  debugInfo.style.display = "block";
}

function renderLoop() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  drawMask();
  requestAnimationFrame(renderLoop);
}

async function main() {
  await initCamera();
  renderLoop();
  setInterval(verifyFrame, 1000);
}

main();
