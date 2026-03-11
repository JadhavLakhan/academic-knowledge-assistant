import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output directory for saved chunks
OUTPUT_DIR = "Chunking/Chunking_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# TEXT CHUNKING
# -----------------------------
def chunk_text(text_file):

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_text(text)

    text_chunks = []

    output_path = os.path.join(OUTPUT_DIR, "text_chunks.txt")

    with open(output_path, "w", encoding="utf-8") as outfile:

        for i, chunk in enumerate(chunks):

            text_chunks.append({
                "chunk_id": f"text_{i}",
                "type": "text",
                "content": chunk
            })

            outfile.write(f"\n------ TEXT CHUNK {i} ------\n")
            outfile.write(chunk)
            outfile.write("\n")

    print(f"Text chunks saved → {output_path}")

    return text_chunks


# -----------------------------
# TABLE CHUNKING
# -----------------------------
def chunk_tables(table_paths):

    table_chunks = []
    seen_tables = set()

    output_path = os.path.join(OUTPUT_DIR, "table_chunks.txt")

    with open(output_path, "w", encoding="utf-8") as outfile:

        for i, path in enumerate(sorted(table_paths)):

            file = os.path.basename(path)

            with open(path, "r", encoding="utf-8") as f:
                table = f.read()

            # remove duplicate tables
            if table in seen_tables:
                continue

            seen_tables.add(table)

            page = file.split("_")[0].replace("page", "")
            table_no = file.split("_")[1].replace("table", "").replace(".md", "")

            # convert markdown table → semantic text
            lines = table.split("\n")
            clean_lines = []

            for line in lines:

                line = line.strip("|")

                if line.strip() == "":
                    continue

                columns = [col.strip() for col in line.split("|")]

                clean_lines.append(" - ".join(columns))

            semantic_table = "\n".join(clean_lines)

            table_chunks.append({
                "chunk_id": f"table_{i}",
                "type": "table",
                "content": semantic_table,
                "document": "GSDP.pdf",
                "page": int(page),
                "table_number": int(table_no),
                "source_file": file
            })

            outfile.write(f"\n------ TABLE {i} ({file}) ------\n")
            outfile.write(semantic_table)
            outfile.write("\n")

    print(f"Improved table chunks saved → {output_path}")

    return table_chunks


# -----------------------------
# IMAGE CAPTION GENERATION
# -----------------------------
def generate_image_caption(image_path):

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a short academic caption describing this chart or figure."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100
    )

    return response.choices[0].message.content


# -----------------------------
# IMAGE CHUNKING
# -----------------------------
def chunk_images(image_paths):

    image_chunks = []

    output_path = os.path.join(OUTPUT_DIR, "image_captions.txt")

    with open(output_path, "w", encoding="utf-8") as outfile:

        for i, img_path in enumerate(image_paths):

            file = os.path.basename(img_path)

            caption = generate_image_caption(img_path)

            page = file.split("_")[0].replace("page", "")

            image_chunks.append({
                "chunk_id": f"image_{i}",
                "type": "image",
                "content": caption,
                "document": "GSDP.pdf",
                "page": int(page),
                "source_file": file
            })

            outfile.write(f"\n------ IMAGE {i} ({file}) ------\n")
            outfile.write(caption)
            outfile.write("\n")

    print(f"Image captions saved → {output_path}")

    return image_chunks


# -----------------------------
# CHUNKING PIPELINE
# -----------------------------
def run_chunking_pipeline(extraction_results):

    text_file = extraction_results["text"]
    table_paths = extraction_results["tables"]
    image_paths = extraction_results["images"]

    print("\nStarting Chunking Pipeline\n")

    text_chunks = chunk_text(text_file)
    table_chunks = chunk_tables(table_paths)
    image_chunks = chunk_images(image_paths)

    all_chunks = text_chunks + table_chunks + image_chunks

    print("Text Chunks:", len(text_chunks))
    print("Table Chunks:", len(table_chunks))
    print("Image Chunks:", len(image_chunks))
    print("Total Chunks:", len(all_chunks))

    return all_chunks


# -----------------------------
# TEST PIPELINE
# -----------------------------
if __name__ == "__main__":

    text_file = "Data_Extraction/text_extraction/GSDP.txt"

    table_dir = "Data_Extraction/table_extract/tables_md"
    image_dir = "Data_Extraction/image_extraction"

    # collect all table files
    table_paths = [
        os.path.join(table_dir, f)
        for f in os.listdir(table_dir)
        if f.endswith(".md")
    ]

    # collect all image files
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    extraction_results = {
        "text": text_file,
        "tables": table_paths,
        "images": image_paths
    }

    chunks = run_chunking_pipeline(extraction_results)

    print("\nPipeline executed successfully.")