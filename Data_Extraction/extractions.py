import fitz
import pdfplumber
import pandas as pd
import os
import re



# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    output_file = "Data_Extraction/text_extraction/GSDP.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_text = ""

    for page in doc:
        text = page.get_text("text")   # better extraction
        all_text += text + "\n"

    # -------- TEXT CLEANING --------

    # remove extra spaces
    all_text = re.sub(r"[ \t]+", " ", all_text)

    # merge broken lines
    all_text = re.sub(r"(?<!\n)\n(?!\n)", " ", all_text)

    # remove page numbers
    all_text = re.sub(r"\n\d+\n", "\n", all_text)

    # remove multiple blank lines
    all_text = re.sub(r"\n{2,}", "\n\n", all_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"Text extracted → {output_file}")

    return output_file


# -----------------------------
# TABLE EXTRACTION
# -----------------------------

def extract_tables(pdf_path):

    OUTPUT_DIR = "Data_Extraction/table_extract"
    TABLE_DIR = os.path.join(OUTPUT_DIR, "tables_md")

    os.makedirs(TABLE_DIR, exist_ok=True)

    table_paths = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages, start=1):

            tables = page.extract_tables()
            print(f"Page {page_num}: Found {len(tables)} tables")

            for t_index, table in enumerate(tables, start=1):

                df = pd.DataFrame(table)
                df = df.dropna(how="all")

                markdown_table = df.to_markdown(index=False)

                md_path = os.path.join(
                    TABLE_DIR, f"page{page_num}_table{t_index}.md"
                )

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(markdown_table)

                table_paths.append(md_path)

                print(f"Saved → {md_path}")

    return table_paths


# -----------------------------
# IMAGE EXTRACTION
# -----------------------------

def extract_images(pdf_path):

    doc = fitz.open(pdf_path)

    OUTPUT_DIR = "Data_Extraction/image_extraction"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MIN_WIDTH = 200
    MIN_HEIGHT = 200
    MIN_BYTES = 10000

    image_paths = []
    saved = 0

    for i in range(len(doc)):

        page = doc[i]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):

            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            width = base_image["width"]
            height = base_image["height"]
            size = len(image_bytes)

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            if size < MIN_BYTES:
                continue

            ext = base_image["ext"]

            filename = f"page{i+1}_img{saved+1}.{ext}"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

            image_paths.append(filepath)

            saved += 1

            print(f"Saved → {filepath}")

    return image_paths


# -----------------------------
# EXTRACTION PIPELINE
# -----------------------------

def run_extraction_pipeline(pdf_path):

    print("\nStarting Extraction Pipeline\n")

    text = extract_text(pdf_path)
    tables = extract_tables(pdf_path)
    images = extract_images(pdf_path)

    print("\nExtraction Completed")

    return {
        "text": text,
        "tables": tables,
        "images": images
    }


# -----------------------------
# RUN FILE DIRECTLY
# -----------------------------

if __name__ == "__main__":

    pdf_path = "Data/GSDP.pdf"

    results = run_extraction_pipeline(pdf_path)

    print(results)