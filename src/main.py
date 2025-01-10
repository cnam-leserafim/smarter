import os
import random
import shutil
import zipfile
from glob import glob

import yaml
from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO

load_dotenv()

# PICSELLIA
WORKSPACE_NAME = "Picsalex-MLOps"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"

# FILE_PATHS
INPUT_DIR = "./input"
INPUT_IMAGES_DIR = f"{INPUT_DIR}/images"
INPUT_ANNOTATIONS_DIR = f"{INPUT_DIR}/annotations"
DATASET_DIR = "./datasets"
IMAGES_DIR = "./images"
LABELS_DIR = "./labels"
YAML_PATH = f"{DATASET_DIR}/config.yaml"

# MODEL
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
random.seed(42)
YOLO_MODEL = "yolo11n.pt"


# Connect to the Picsellia client
def connect_to_client():
    return Client(
        api_token=os.getenv("PICSELLIA_API_TOKEN"),
        organization_name=WORKSPACE_NAME,
    )


# Downloading the dataset from Picsellia
def import_datasets(client):
    dataset = client.get_dataset_version_by_id(DATASET_ID)
    os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
    dataset.list_assets().download(INPUT_IMAGES_DIR)
    print("Imported datasets")
    return dataset


def get_experiment(client):
    # Existing experiment
    project = client.get_project(project_name="Groupe_1")
    experiment = project.get_experiment(name="experiment-0")
    print(f"Existing experimentation recovered : {experiment.name}")
    datasets = experiment.list_attached_dataset_versions()
    print(f"Datasets attached to the experience : {datasets}")
    """
        experiment = project.create_experiment(
            name="experiment-1",
            description="base experiment",
        )

        experiment.attach_dataset(
            name="⭐️ cnam_product_2024",
            dataset_version=dataset,
        )
        print(f"Creation of new experiment : {experiment.name}")
        """


# Export of annotations in YOLO format
def export_annotations(dataset):
    os.makedirs(INPUT_ANNOTATIONS_DIR, exist_ok=True)
    dataset.export_annotation_file(
        AnnotationFileType.YOLO, INPUT_ANNOTATIONS_DIR
    )
    print(f"Annotations exported to : {INPUT_ANNOTATIONS_DIR}")


def extract_annotations():
    # Find the first ZIP archive in the folder or subfolder
    zip_file = next(
        (
            os.path.join(root, file)
            for root, _, files in os.walk(INPUT_ANNOTATIONS_DIR)
            for file in files
            if file.endswith(".zip")
        ),
        None,
    )
    if zip_file:
        # Create the "annotations" folder if it does not exist
        os.makedirs(INPUT_ANNOTATIONS_DIR, exist_ok=True)

        # Unzip the ZIP archive
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(INPUT_ANNOTATIONS_DIR)
        print(f"Archive unzipped in : {INPUT_ANNOTATIONS_DIR}")

        # TODO : Delete parent folders hash/annotations/
        # Delete ZIP archive
        os.remove(zip_file)
        print(f"Archive {zip_file} deleted.")
    else:
        print(f"No ZIP archive found in {INPUT_DIR} folder or its subfolders.")

    # Check extracted files
    extracted_files = os.listdir(INPUT_ANNOTATIONS_DIR)
    print(f"Extracted files: {extracted_files}")
    print(f"Extracted files: {extracted_files}")
    file_count = len(extracted_files)
    print(f"Total number of extracted files : {file_count}")


# Split data into train, validation, and test sets
def split_data():
    # List of images and labels
    all_images = glob(f"{INPUT_IMAGES_DIR}/*.jpg")
    all_labels = glob(f"{INPUT_ANNOTATIONS_DIR}/*.txt")

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
def copy_files(pairs, dest_image_dir, dest_label_dir):
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    for image, label in pairs:
        shutil.copy(image, dest_image_dir)
        shutil.copy(label, dest_label_dir)


# Generating the YAML file
def generate_yaml_file():
    config = {
        "train": f"{os.path.abspath('datasets/train/images')}",
        "val": f"{os.path.abspath('datasets/val/images')}",
        "test": f"{os.path.abspath('datasets/test/images')}",
        "nc": 10,
        "names": [
            "Canettes",
            "Bouteilles en plastique",
            "Pepito",
            "Kinder Country",
            "Kinder Tronky",
            "Kinder Pinguy",
            "Tic-Tac",
            "Sucette",
            "Capsule",
            "Mikado",
        ],
    }
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(YAML_PATH, "w") as yaml_file:
        yaml.dump(config, yaml_file)
    print(f"Generated config.yaml file : {YAML_PATH}")


def main():
    # --- PART 1 : Import images and annotations ---
    client = connect_to_client()
    dataset = import_datasets(client)
    get_experiment(client)
    export_annotations(dataset)
    extract_annotations()

    # --- PART 2 : Split data for Ultralytics YOLO ---
    split_data_dict = split_data()
    # Copying files to corresponding directories
    for split, pairs in split_data_dict.items():
        copy_files(
            pairs,
            f"{DATASET_DIR}/{split}/{IMAGES_DIR}",
            f"{DATASET_DIR}/{split}/{LABELS_DIR}",
        )
    generate_yaml_file()

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(YOLO_MODEL)

    # Train the model using the dataset
    results = model.train(
        data=YAML_PATH, epochs=30, lr0=0.001, batch=16, plots=True
    )

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    # success = model.export(format="onnx")

    print(results)


if __name__ == "__main__":
    main()
