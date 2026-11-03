from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, send_from_directory
)
from pathlib import Path
import subprocess
import sys
import os
import shutil
import fitz
import cv2
import numpy as np





def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

os.makedirs("Input", exist_ok=True)
os.makedirs("Output", exist_ok=True)
os.makedirs("OverlayComparison", exist_ok=True)

# app = Flask(__name__)

#
# app = Flask(
#     __name__,
#     template_folder="templates",
#     static_folder="static")

app = Flask(
    __name__,
    template_folder=resource_path("templates")
)

app.secret_key = "change-this-secret-key"

# Project folders
# # BASE_DIR = Path(__file__).resolve().parent
# def resource_path(relative_path=""):
#     try:
#         base_path = Path(sys._MEIPASS)
#     except AttributeError:
#         base_path = Path(__file__).resolve().parent
#
#     return base_path / relative_path



if getattr(sys, 'frozen', False):
    BASE_DIR = Path(os.path.dirname(sys.executable))
else:
    BASE_DIR = Path(__file__).resolve().parent




INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
MARKED_DIR = BASE_DIR / "OverlayComparison"

# Hidden metadata stores the main file's original filename.
MAIN_META = INPUT_DIR / ".main_filename"

for folder in (INPUT_DIR, OUTPUT_DIR, MARKED_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def clean_filename(filename):
    """Keep the original filename while preventing path traversal."""
    if not filename:
        return ""

    # Remove any directory components, for Windows and other systems.
    filename = filename.replace("\\", "/").split("/")[-1]

    if filename in ("", ".", "..") or "\x00" in filename:
        return ""

    if filename == MAIN_META.name:
        return ""

    return filename


def get_main_filename():
    if not MAIN_META.is_file():
        return None

    name = MAIN_META.read_text(encoding="utf-8").strip()
    name = clean_filename(name)

    if name and (INPUT_DIR / name).is_file():
        return name

    return None


def get_sub_files():
    main_name = get_main_filename()

    return sorted(
        [
            f for f in INPUT_DIR.iterdir()
            if f.is_file()
               and f.name != MAIN_META.name
               and f.name != main_name
        ],
        key=lambda f: f.name.lower()
    )


def clear_folder(folder):
    """Delete folder contents without deleting the folder itself."""
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_symlink() or item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def pdf_to_image(pdf_path, page_num):

    doc = fitz.open(str(pdf_path))

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

    clear_folder(MARKED_DIR)

    os.makedirs(MARKED_DIR, exist_ok=True)

    doc1 = fitz.open(str(pdf1))
    doc2 = fitz.open(str(pdf2))

    pages = min(
        len(doc1),
        len(doc2)
    )

    pdf1_name = pdf1.stem
    pdf2_name = pdf2.stem

    for page_num in range(pages):

        page_folder = MARKED_DIR / f"Page_{page_num + 1}"

        page_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        img1 = pdf_to_image(
            str(pdf1),
            page_num
        )

        img2 = pdf_to_image(
            str(pdf2),
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

        removed_mask = cv2.bitwise_and(
            mask1,
            cv2.bitwise_not(mask2)
        )

        added_mask = cv2.bitwise_and(
            mask2,
            cv2.bitwise_not(mask1)
        )

        combined = img1.copy()

        combined[removed_mask > 0] = [255, 0, 0]

        combined[added_mask > 0] = [0, 0, 255]

        pdf1_marked = img1.copy()

        pdf1_marked[removed_mask > 0] = [255, 0, 0]

        pdf2_marked = img2.copy()

        pdf2_marked[added_mask > 0] = [0, 0, 255]

        cv2.imwrite(
            str(page_folder / "Combined.png"),
            combined
        )

        cv2.imwrite(
            str(page_folder / f"{pdf1_name}_Diff.png"),
            pdf1_marked
        )

        cv2.imwrite(
            str(page_folder / f"{pdf2_name}_Diff.png"),
            pdf2_marked
        )

    doc1.close()
    doc2.close()

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    main_name = get_main_filename()
    sub_files = get_sub_files()
    # pdf_files = sorted(
    #     MARKED_DIR.glob("*.pdf"),
    #     key=lambda f: f.name.lower()
    # )
    #
    # return render_template(
    #     "index.html",
    #     main_filename=main_name,
    #     sub_files=[f.name for f in sub_files],
    #     pdf_files=pdf_files
    # )
    pages = []

    for folder in sorted(os.listdir(MARKED_DIR)):
        page_path = MARKED_DIR / folder

        if not page_path.is_dir():
            continue

        files = sorted(os.listdir(page_path))

        combined = None
        others = []

        for f in files:
            if f.lower() == "combined.png":
                combined = f
            else:
                others.append(f)

        if len(others) >= 2:
            pages.append({
                "folder": folder,
                "left": others[0],
                "right": others[1],
                "combined": combined
            })

    return render_template(
        "index.html",
        main_filename=main_name,
        sub_files=[f.name for f in sub_files],
        pages=pages
    )

# --------------------------------------------------
# UPLOAD MAIN FILE
# --------------------------------------------------

@app.route("/upload-main", methods=["POST"])
def upload_main():
    file = request.files.get("main_file")

    if not file or not file.filename:
        flash("Please select a main file.", "error")
        return redirect(url_for("home"))

    filename = clean_filename(file.filename)

    if not filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("home"))

    old_main = get_main_filename()
    save_path = INPUT_DIR / filename

    # Write the new file before updating the main-file record.
    try:
        file.save(str(save_path))

        # Remove the previous main file only if its name changed.
        if old_main and old_main != filename:
            old_path = INPUT_DIR / old_main
            if old_path.is_file():
                old_path.unlink()

        MAIN_META.write_text(filename, encoding="utf-8")

    except OSError as error:
        app.logger.exception("Main file upload failed")
        flash(f"Could not save main file: {error}", "error")
        return redirect(url_for("home"))

    app.logger.info("Main file saved: %s", save_path)
    flash(f"Main file saved: {filename}", "success")

    return redirect(url_for("home"))


# --------------------------------------------------
# UPLOAD SUB FILES DIRECTLY INTO INPUT
# --------------------------------------------------

@app.route("/upload-subs", methods=["POST"])
def upload_subs():
    files = request.files.getlist("sub_files")

    if not any(f and f.filename for f in files):
        flash("Please select at least one sub file.", "error")
        return redirect(url_for("home"))

    main_name = get_main_filename()
    saved = 0
    skipped = []

    for file in files:
        if not file or not file.filename:
            continue

        filename = clean_filename(file.filename)

        if not filename:
            skipped.append("Invalid filename")
            continue

        # Never overwrite the main file with a sub file.
        if filename == main_name:
            skipped.append(filename)
            continue

        save_path = INPUT_DIR / filename

        try:
            file.save(str(save_path))
            saved += 1
            app.logger.info("Sub file saved: %s", save_path)
        except OSError:
            app.logger.exception("Could not save sub file: %s", filename)
            skipped.append(filename)

    if saved:
        flash(f"{saved} sub file(s) saved directly in Input.", "success")

    if skipped:
        flash(
            "Skipped files (invalid names or matching the main filename): "
            + ", ".join(skipped),
            "error"
        )

    return redirect(url_for("home"))


# --------------------------------------------------
# CLEAR SUB FILES + OUTPUT + MARKED
# --------------------------------------------------

@app.route("/clear-subs", methods=["POST"])
def clear_subs():
    main_name = get_main_filename()
    deleted_subs = 0

    # Remove sub files only; preserve the main file and metadata.
    try:
        for file in get_sub_files():
            file.unlink()
            deleted_subs += 1

        # Clear generated files in both folders.
        clear_folder(OUTPUT_DIR)
        clear_folder(MARKED_DIR)

        flash(
            f"Cleared {deleted_subs} sub file(s), Output, and Marked. "
            "Your main file is preserved.",
            "success"
        )

    except OSError as error:
        app.logger.exception("Clear operation failed")
        flash(f"Could not clear all files: {error}", "error")

    return redirect(url_for("home"))


# --------------------------------------------------
# EXECUTE PYTHON SCRIPTS
# --------------------------------------------------

def run_script(script_name):
    script_path = BASE_DIR / script_name

    if not script_path.is_file():
        flash(
            f"{script_name} not found. Expected: {script_path}",
            "error"
        )
        return redirect(url_for("home"))

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.stdout:
            print(f"--- {script_name} output ---")
            print(result.stdout)

        if result.stderr:
            print(f"--- {script_name} errors ---")
            print(result.stderr)

        if result.returncode == 0:
            flash(f"{script_name} executed successfully.", "success")
        else:
            flash(
                f"{script_name} failed with exit code "
                f"{result.returncode}. Check the PyCharm terminal.",
                "error"
            )

    except subprocess.TimeoutExpired:
        flash(f"{script_name} timed out after 5 minutes.", "error")

    except OSError as error:
        flash(f"Could not execute {script_name}: {error}", "error")

    return redirect(url_for("home"))


# @app.route("/run-main", methods=["POST"])
# def run_main():
#     return run_script("main.py")

# @app.route("/page_image/<folder>/<filename>")
# def page_image(folder, filename):
#     return send_from_directory(
#         MARKED_DIR / folder,
#         filename
#     )

@app.route("/page_image/<folder>/<filename>")
def page_image(folder, filename):
    return send_from_directory(
        MARKED_DIR / folder,
        filename
    )


# @app.route("/run-marking", methods=["POST"])
# def run_marking():
#     return run_script("Pixel_comparison.py")

@app.route("/run-marking", methods=["POST"])
def run_marking():

    try:

        main_file = get_main_filename()

        if not main_file:
            flash("Please upload a main PDF first.", "error")
            return redirect(url_for("home"))

        main_pdf = INPUT_DIR / main_file

        sub_files = [
            f for f in get_sub_files()
            if f.suffix.lower() == ".pdf"
        ]

        if not sub_files:
            flash("Please upload at least one sub PDF.", "error")
            return redirect(url_for("home"))

        compare_pdfs_overlay(
            main_pdf,
            sub_files[0]
        )

        flash(
            "Comparison completed successfully.",
            "success"
        )

    except Exception as e:

        app.logger.exception(
            "Comparison failed"
        )

        flash(
            f"Comparison failed: {e}",
            "error"
        )

    return redirect(url_for("home"))


# --------------------------------------------------
# VIEW GENERATED PDFS
# --------------------------------------------------

@app.route("/pdf/<path:filename>")
def view_pdf(filename):
    return send_from_directory(MARKED_DIR, filename)


# --------------------------------------------------
# CLEAR EVERYTHING
# --------------------------------------------------

@app.route("/clear-all", methods=["POST"])
def clear_all():
    try:
        clear_folder(INPUT_DIR)
        clear_folder(OUTPUT_DIR)
        clear_folder(MARKED_DIR)

        flash("All input files and generated outputs have been cleared.",
              "success")

    except OSError as error:
        app.logger.exception("Clear all failed")
        flash(f"Could not clear all files: {error}", "error")

    return redirect(url_for("home"))


if __name__ == "__main__":
    print("Project folder:", BASE_DIR)
    print("Input folder:", INPUT_DIR)
    print("Output folder:", OUTPUT_DIR)
    print("Marked folder:", MARKED_DIR)

    app.run(debug=True)