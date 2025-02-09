import os
import random
import shutil
import zipfile
from glob import glob

from picsellia import DatasetVersion
from picsellia.types.enums import AnnotationFileType

SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


# Downloading the dataset from Picsellia
def download_dataset(dest_dir: str, dataset: DatasetVersion) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    dataset.list_assets().download(dest_dir)
    print("Imported dataset")


# Export of annotations in YOLO format
def export_annotations(dataset: DatasetVersion, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    dataset.export_annotation_file(AnnotationFileType.YOLO, dest_dir)
    print(f"Annotations exported to : {dest_dir}")


def extract_annotations(dest_dir: str) -> None:
    # Find the first ZIP archive in the folder or subfolder
    zip_file = next(
        (
            os.path.join(root, file)
            for root, _, files in os.walk(dest_dir)
            for file in files
            if file.endswith(".zip")
        ),
        None,
    )
    if zip_file:
        # Create the "annotations" folder if it does not exist
        os.makedirs(dest_dir, exist_ok=True)

        # Unzip the ZIP archive
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Archive unzipped in : {dest_dir}")

        # TODO : Delete parent folders hash/annotations/
        # Delete ZIP archive
        os.remove(zip_file)
        print(f"Archive {zip_file} deleted.")
    else:
        print(f"No ZIP archive found in {dest_dir} folder or its subfolders.")

    # Check extracted files
    extracted_files = os.listdir(dest_dir)
    print(f"Extracted files: {extracted_files}")
    print(f"Extracted files: {extracted_files}")
    file_count = len(extracted_files)
    print(f"Total number of extracted files : {file_count}")


# Split data into train, validation, and test sets
def get_split_data(images_dir: str, annotations_dir: str) -> dict:
    # List of images and labels
    all_images = glob(f"{images_dir}/*.jpg")
    all_labels = glob(f"{annotations_dir}/*.txt")

    # Associate images and labels
    data_pairs = list(zip(all_images, all_labels))
    random.shuffle(data_pairs)

    # Calculation of indices for each split
    train_idx = int(len(data_pairs) * SPLIT_RATIOS["train"])
    val_idx = train_idx + int(len(data_pairs) * SPLIT_RATIOS["val"])

    return {
        "train": data_pairs[:train_idx],
        "val": data_pairs[train_idx:val_idx],
        "test": data_pairs[val_idx:],
    }


# Copying files to respective directories
def copy_files(pairs, dest_image_dir, dest_label_dir) -> None:
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    for image, label in pairs:
        shutil.copy(image, dest_image_dir)
        shutil.copy(label, dest_label_dir)
