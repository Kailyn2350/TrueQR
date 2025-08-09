import os
import math
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import pagesizes
from reportlab.lib.units import cm


def create_printable_pdf(image_folder, output_pdf_path):
    """
    Scans a folder for images and arranges them in a grid on a single A4 PDF page.
    """
    # --- 1. Setup ---
    image_paths = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if not image_paths:
        print(f"No images found in '{image_folder}'.")
        return

    # --- 2. PDF and Layout Configuration ---
    page_width, page_height = pagesizes.A4
    margin = 1 * cm
    content_width = page_width - 2 * margin
    content_height = page_height - 2 * margin

    num_images = len(image_paths)
    # Calculate grid size to be as square as possible
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))

    # Calculate the size of each cell in the grid
    cell_width = content_width / cols
    cell_height = content_height / rows

    # Determine the image size to fit within the cell, maintaining aspect ratio
    # We'll use the smaller of the two cell dimensions as the max size
    image_max_size = min(cell_width, cell_height) * 0.9  # Use 90% of cell for padding

    print(f"Found {num_images} images. Creating a {cols}x{rows} grid on an A4 page.")
    print(f"Output PDF: {output_pdf_path}")

    # --- 3. PDF Generation ---
    c = canvas.Canvas(output_pdf_path, pagesize=pagesizes.A4)

    for i, image_path in enumerate(image_paths):
        row = i // cols
        col = i % cols

        # Calculate position of the center of the cell
        cell_center_x = margin + (col * cell_width) + (cell_width / 2)
        cell_center_y = page_height - margin - (row * cell_height) - (cell_height / 2)

        # Open image to get its aspect ratio
        with Image.open(image_path) as img:
            aspect = img.height / img.width

            # Calculate drawing dimensions
            draw_w = image_max_size
            draw_h = image_max_size * aspect
            if draw_h > image_max_size:
                draw_h = image_max_size
                draw_w = image_max_size / aspect

        # Calculate bottom-left corner for drawing
        draw_x = cell_center_x - (draw_w / 2)
        draw_y = cell_center_y - (draw_h / 2)

        c.drawImage(
            image_path,
            draw_x,
            draw_y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            anchor="c",
        )

    c.save()
    print("PDF generation complete.")


if __name__ == "__main__":
    # Assuming the script is in the TrueQR folder
    project_root = os.path.dirname(os.path.abspath(__file__))
    secured_folder = os.path.join(project_root, "secured")
    output_file = os.path.join(project_root, "printable_qr_sheet.pdf")

    create_printable_pdf(secured_folder, output_file)
