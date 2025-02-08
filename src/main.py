import os
import random
import shutil
import time
import zipfile
from glob import glob

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from PicselliaClient import PicselliaClient
from PicselliaLogger import PicselliaLogger

load_dotenv()

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
    client = PicselliaClient()
    dataset = client.import_datasets(INPUT_IMAGES_DIR)
    client.import_experiment()
    client.export_annotations(dataset, INPUT_ANNOTATIONS_DIR)
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
    model = YOLO("best.pt")

    # Add callbacks for logs
    logger: PicselliaLogger = PicselliaLogger(client.get_experiment())
    model.add_callback("on_train_start", logger.on_train_start)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_train_end", logger.on_train_end)

    # Train the model using the dataset
    # results =
    model.train(
        data=YAML_PATH,
        epochs=2,
        lr0=0.001,
        batch=16,
        patience=10,
        imgsz=640,
        plots=True,
        close_mosaic=0,
    )

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    # success = model.export(format="onnx")

    model_name = "smarter" + time.strftime("-%Y-%m-%d-%H-%M-%S")
    model_version = client.create_model_version(model_name)

    client.get_experiment().attach_model_version(model_version, True)

    model_version.store("model-latest", model.trainer.best, do_zip=True)

    print(results)


if __name__ == "__main__":
    main()
