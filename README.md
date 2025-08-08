# TrueQR: A Project for QR Code Forgery Detection

## Abstract

This project's primary goal was to develop a logic capable of distinguishing between an original, secured QR code and a counterfeit version created by scanning or copying. The core idea was to embed a fragile "encryption" or digital watermark into the original QR code image. This watermark is designed to be destroyed or significantly altered during the copying process, making forgeries detectable.

This document outlines the methodology, the success in a controlled digital environment, and the challenges discovered when applying the technique to real-world scenarios involving physical media and cameras.

## Core Concepts & Methodology

The verification process was not about finding generic artifacts of copying, but about breaking a specific, embedded signal. The original, genuine QR code has a unique signature embedded within it using a combination of the following techniques:

1.  **Gaussian Noise:** A specific, low-amplitude noise pattern is added.
2.  **Frequency Domain Manipulation:** The image is transformed to the frequency domain to subtly alter its high and low-frequency components.
3.  **Fourier Phase Transformation:** The phase of the image's Fourier transform is modified, embedding information that is sensitive to translation and rotation.

The combination of these techniques creates a QR code that is **visually indistinguishable** from a standard one. However, when a counterfeit is made (e.g., by taking a screenshot or printing and scanning), the fragile embedded signature is broken. The verification logic then compares the signature of a suspect image to the expected signature of the original.

## Outcomes and Key Findings

### 1. Success in Digital PNG-to-PNG Comparison

**This method was highly successful.** When comparing an "encrypted" source PNG file with a "forged" PNG file (e.g., a screenshot of the original), the system could reliably and accurately differentiate between the genuine and the counterfeit. This confirmed that the core principle of using a fragile, breakable watermark is valid in a purely digital domain where environmental variables are eliminated. To the naked eye, the two PNG files appear virtually identical.

### 2. The Challenge of Physical Media

The project's primary challenge emerged when moving from digital files to real-world application. When attempting to verify a printed QR code using a smartphone camera (iPhone 13 Pro), the results became inconsistent.

**Reason for Inconsistency:** The verification logic was so sensitive that it was affected by the physical properties of the medium itself. The inference results changed based on:
*   The texture, gloss, and color of the paper.
*   Ambient lighting conditions.
*   The specific angle and distance of the camera.

This means that for the system to work reliably, it requires a **highly controlled environment**. For example, verification is possible if the QR code is always printed on a specific, standardized type of paper under controlled lighting. This limits the universal applicability of the method but proves its viability for high-security scenarios where the printing medium can be standardized.

### 3. The Discovery of Display-Based Issues

During this process, an additional complication was discovered: verifying a QR code displayed on a screen (like a monitor or phone) with a camera is fundamentally different and more complex than verifying a printed one. The pixel grid, screen brightness, and refresh rate of the display introduce a new set of variables that interfere with the fragile watermark, leading to failed verification. This highlighted the critical need to **limit the material and medium** for the verification to be considered reliable.

## How to Use This Project

### Prerequisites
*   Python 3.x
*   Required Python libraries (e.g., OpenCV, NumPy, scikit-image). You can install them via pip:
    ```bash
    pip install opencv-python numpy scikit-image
    ```
*   For web-based verification: `Flask` and `ngrok`.
    ```bash
    pip install Flask
    ```

### Usage 1: Verifying a PNG File

To verify a local QR code image file, you can run the `test_verify.py` script from your terminal.

**Command:**
```bash
python test_verify.py --image "path/to/your/qrcode.png"
```
The script will analyze the image and output whether it believes the file is genuine or a forgery.

### Usage 2: Web-Based Camera Verification (Experimental)

This setup allows you to use your computer's or phone's camera to attempt real-time verification.

**Note:** As stated above, this method is experimental and likely to fail, but it demonstrates the intended real-world application.

**Step 1: Start the Local Web Server**

The web application is located in the `OCR/` directory. Start the Flask server. It is assumed to run on port 8000.

```bash
cd OCR
python app.py
```

**Step 2: Expose the Server with ngrok**

Modern web browsers require a secure `https://` connection to access camera hardware. `ngrok` is a tool that creates a secure public URL for your local server.

In a **new terminal window**, run the following command:

```bash
ngrok http 8000
```

**Step 3: Access the Application**

`ngrok` will provide you with a public HTTPS URL (e.g., `https://random-string.ngrok.io`). Open this URL in the web browser of the device you want to use for scanning (e.g., your iPhone). You can then grant camera access to the site and attempt to verify a QR code.
