import cv2
import numpy as np
import os
import glob

# --- 설정 ---
INPUT_FOLDER = "False"  # 이미지가 있는 폴더
MAIN_OUTPUT_FOLDER = "grid_output_false"  # 메인 출력 폴더
GRID_SIZE = (3, 3)  # 3x3 격자

# 격자 전체의 시작 위치를 미세 조정합니다 (픽셀 단위)
GRID_OFFSET_X = 10  # 양수: 오른쪽으로 이동
GRID_OFFSET_Y = 5  # 양수: 아래쪽으로 이동
# --- ---

# 입력 폴더에서 모든 .jpg 파일 목록 가져오기
image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))

if not image_paths:
    print(f"Error: No .jpg files found in the '{INPUT_FOLDER}' directory.")
    exit()

print(f"Found {len(image_paths)} images to process.")

# 각 이미지 파일에 대해 반복
for image_path in image_paths:
    print(f"\nProcessing: {image_path}")

    # --- 출력 폴더 설정 ---
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    per_image_output_folder = os.path.join(MAIN_OUTPUT_FOLDER, base_filename)

    if not os.path.exists(per_image_output_folder):
        os.makedirs(per_image_output_folder)

    # 이미지 불러오기
    img = cv2.imread(image_path)
    if img is None:
        print(f"  - Error: Could not read image.")
        continue  # 다음 파일로 넘어감

    # 1. 전처리 및 전체 QR 영역 찾기
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  - Error: No contours found.")
        continue

    all_points = np.concatenate([c for c in contours])
    x, y, w, h = cv2.boundingRect(all_points)

    # 원본 이미지에서 전체 QR 영역만 잘라내기 (오프셋 적용)
    crop_x = x + GRID_OFFSET_X
    crop_y = y + GRID_OFFSET_Y
    qr_block = img[crop_y : crop_y + h, crop_x : crop_x + w]

    print(
        f"  - QR Code block found. Slicing into {GRID_SIZE[0]}x{GRID_SIZE[1]} grid..."
    )

    # 2. 격자로 자르기
    block_h, block_w, _ = qr_block.shape
    cell_h = block_h // GRID_SIZE[0]
    cell_w = block_w // GRID_SIZE[1]

    count = 0
    for i in range(GRID_SIZE[0]):  # Rows
        for j in range(GRID_SIZE[1]):  # Columns
            count += 1
            cell_x_start = j * cell_w
            cell_y_start = i * cell_h
            cell_x_end = cell_x_start + cell_w
            cell_y_end = cell_y_start + cell_h

            cell = qr_block[cell_y_start:cell_y_end, cell_x_start:cell_x_end]

            output_filename = os.path.join(
                per_image_output_folder, f"qr_part_{count}.png"
            )
            cv2.imwrite(output_filename, cell)

    print(f"  - Successfully saved {count} parts to '{per_image_output_folder}'")

print("\nAll tasks finished.")
