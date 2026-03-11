import boto3
import os


def download_files_from_s3(bucket_name, download_path):

    s3 = boto3.client("s3")

    os.makedirs(download_path, exist_ok=True)

    response = s3.list_objects_v2(Bucket=bucket_name)

    downloaded_files = []

    for obj in response.get("Contents", []):

        file_key = obj["Key"]

        file_name = os.path.basename(file_key)

        local_file = os.path.join(download_path, file_name)

        print(f"Downloading {file_name}...")

        s3.download_file(
            bucket_name,
            file_key,
            local_file
        )

        downloaded_files.append(local_file)

    print("All files downloaded successfully")

    return downloaded_files