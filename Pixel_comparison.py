import fitz
import cv2
import numpy as np
import os


def pdf_to_image(pdf_path, page_num):

    doc = fitz.open(pdf_path)

    page = doc[page_num]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(5, 5),
        alpha=False
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    doc.close()

    return img


def compare_pdfs_overlay(pdf1, pdf2):

    output_folder = "OverlayComparison"
    os.makedirs(output_folder, exist_ok=True)

    doc1 = fitz.open(pdf1)
    doc2 = fitz.open(pdf2)

    total_pages_pdf1 = len(doc1)
    total_pages_pdf2 = len(doc2)

    pages = min(
        total_pages_pdf1,
        total_pages_pdf2
    )

    print(f"\nPDF1 Pages : {total_pages_pdf1}")
    print(f"PDF2 Pages : {total_pages_pdf2}")
    print(f"Comparing   : {pages}")

    if total_pages_pdf1 != total_pages_pdf2:
        print(
            f"\nWARNING : Page count mismatch "
            f"(PDF1={total_pages_pdf1}, PDF2={total_pages_pdf2})"
        )

    pdf1_name = os.path.splitext(
        os.path.basename(pdf1)
    )[0]

    pdf2_name = os.path.splitext(
        os.path.basename(pdf2)
    )[0]

    for page_num in range(pages):

        print(
            f"\nProcessing Page "
            f"{page_num + 1}/{pages}"
        )

        page_folder = os.path.join(
            output_folder,
            f"Page_{page_num + 1}"
        )

        os.makedirs(
            page_folder,
            exist_ok=True
        )

        img1 = pdf_to_image(
            pdf1,
            page_num
        )

        img2 = pdf_to_image(
            pdf2,
            page_num
        )

        img2 = cv2.resize(
            img2,
            (
                img1.shape[1],
                img1.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )

        _, mask1 = cv2.threshold(
            gray1,
            240,
            255,
            cv2.THRESH_BINARY_INV
        )

        _, mask2 = cv2.threshold(
            gray2,
            240,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Present only in PDF1
        removed_mask = cv2.bitwise_and(
            mask1,
            cv2.bitwise_not(mask2)
        )

        # Present only in PDF2
        added_mask = cv2.bitwise_and(
            mask2,
            cv2.bitwise_not(mask1)
        )

        # -----------------------------
        # Combined Image
        # -----------------------------
        combined = img1.copy()

        combined[
            removed_mask > 0
        ] = [255, 0, 0]

        combined[
            added_mask > 0
        ] = [0, 0, 255]

        # -----------------------------
        # PDF1 Only
        # -----------------------------
        pdf1_marked = img1.copy()

        pdf1_marked[
            removed_mask > 0
        ] = [255, 0, 0]

        # -----------------------------
        # PDF2 Only
        # -----------------------------
        pdf2_marked = img2.copy()

        pdf2_marked[
            added_mask > 0
        ] = [0, 0, 255]

        # -----------------------------
        # File Names
        # -----------------------------
        combined_file = os.path.join(
            page_folder,
            "Combined.png"
        )

        pdf1_file = os.path.join(
            page_folder,
            f"{pdf1_name}_Diff.png"
        )

        pdf2_file = os.path.join(
            page_folder,
            f"{pdf2_name}_Diff.png"
        )

        # -----------------------------
        # Save
        # -----------------------------
        cv2.imwrite(
            combined_file,
            combined
        )

        cv2.imwrite(
            pdf1_file,
            pdf1_marked
        )

        cv2.imwrite(
            pdf2_file,
            pdf2_marked
        )

        print(
            f"Removed Pixels : "
            f"{np.count_nonzero(removed_mask)}"
        )

        print(
            f"Added Pixels   : "
            f"{np.count_nonzero(added_mask)}"
        )

    doc1.close()
    doc2.close()

    print(
        f"\nResults saved in '{output_folder}'"
    )


# ======================================
# INPUT PDFs FROM INPUT FOLDER
# ======================================

input_folder = "Input"

pdf_files = sorted([
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".pdf")
])

if len(pdf_files) < 2:

    print(
        "Place at least two PDFs "
        "inside Input folder."
    )

else:

    pdf1 = pdf_files[0]
    pdf2 = pdf_files[1]

    print(
        f"\nPDF1 : "
        f"{os.path.basename(pdf1)}"
    )

    print(
        f"PDF2 : "
        f"{os.path.basename(pdf2)}"
    )

    compare_pdfs_overlay(
        pdf1,
        pdf2
    )