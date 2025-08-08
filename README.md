# TrueQR: A Project for QR Code Forgery Detection

## Abstract

This project explores various methodologies for detecting counterfeit QR codes. The primary goal is to differentiate between an original, authentic QR code and one that has been copied, for example, through a screenshot or a physical scan. The project investigates both digital image analysis and real-time camera-based verification, documenting the challenges and successes of each approach.

## Core Concepts & Initial Hypothesis

The initial approach was based on the idea that the process of copying a QR code—whether through digital screen capture or physical printing and scanning—introduces subtle, yet detectable, artifacts. The plan was to identify these artifacts using a combination of image processing techniques:

1.  **Gaussian Noise Analysis:** Counterfeit images, especially from scans, often exhibit a different noise profile compared to the clean, digitally generated original.
2.  **High and Low-Frequency Analysis:** The sharp edges of a genuine QR code have distinct high-frequency components. Copying can blur these edges, altering the frequency distribution.
3.  **Fourier Phase Transformation:** The phase spectrum of an image is sensitive to structural changes. The hypothesis was that forgery would alter the phase information in a measurable way.

### Outcome: A Challenging Reality

This initial methodology **failed** to produce reliable results. The system was unable to consistently distinguish between genuine and forged QR codes when dealing with images captured from the physical world.

**Reason for Failure:** The environmental conditions of the QR code had a greater impact on the image analysis than the forgery process itself. Factors like:
*   The type of display showing the QR code
*   The quality and texture of the paper it was printed on
*   The lighting conditions during the scan or photo
*   The angle and quality of the camera

...all introduced variables that a simple algorithmic approach could not overcome.

## Successful Approaches & Current Status

While the physical capture method was unreliable, the project found success in other areas.

### 1. Digital PNG File Verification

**This method is successful.** When analyzing the raw digital PNG files, the system can reliably and accurately distinguish between a genuine, digitally-created QR code and a forged one (e.g., a screenshot that was saved as a new PNG). This proves that the underlying principle of artifact detection is valid in a controlled, digital-only environment.

### 2. Real-time Webcam Verification

**This method is currently unsuccessful.** An attempt was made to use a high-quality smartphone camera (iPhone 13 Pro) to verify a QR code displayed on a screen. This failed for reasons similar to the physical print-and-scan method. The display environment (screen brightness, pixel grid, refresh rate, viewing angle) is fundamentally different from a static paper environment, and the system was not able to account for these variations.

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
