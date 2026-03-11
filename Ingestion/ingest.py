from Ingestion.s3_loader import download_files_from_s3
from Data_Extraction.extractions import run_extraction_pipeline
from Chunking.chunking_stratergies import run_chunking_pipeline
from Embedding.vector_store import run_embedding_pipeline


def run_ingestion_pipeline(bucket_name):

    print("Starting Ingestion Pipeline\n")

    # Step 1: Download files from S3
    downloaded_files = download_files_from_s3(
        bucket_name=bucket_name,
        download_path="Data"
    )

    print("\nFiles downloaded:", downloaded_files)

    all_chunks = []

    # Step 2: Process each PDF
    for pdf_path in downloaded_files:

        print(f"\nProcessing file: {pdf_path}")

        # Extraction
        extracted_data = run_extraction_pipeline(pdf_path)

        print("\nExtraction Summary")
        print("Text file:", extracted_data["text"])
        print("Tables:", len(extracted_data["tables"]))
        print("Images:", len(extracted_data["images"]))

        # Chunking
        chunks = run_chunking_pipeline(extracted_data)

        print("\nChunking Completed")
        print("Total Chunks:", len(chunks))

        all_chunks.extend(chunks)

    # Step 3: Embedding + Vector Storage
    run_embedding_pipeline(all_chunks)

    return all_chunks


if __name__ == "__main__":

    bucket_name = "ai.academic.rag"

    run_ingestion_pipeline(bucket_name)