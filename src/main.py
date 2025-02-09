import os
import random
import time

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.picsellia_resources.picsellia_client import PicselliaClient
from src.picsellia_resources.picsellia_logger import PicselliaLogger
from src.utils import (
    copy_files,
    download_dataset,
    export_annotations,
    extract_annotations,
    get_split_data,
)

import src.config as config

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


# HYPER PARAMETERS
parameters = {
    "epochs": 150,
    "seed": 42,
    "lr0": 0.001,
    "lrf": 0.003,
    "momentum": 0.96,
    "weight_decay": 0.00048,
    "batch": 16,
    "patience": 30,
    "imgsz": 640,
    "plots": True,
    "close_mosaic": 0,
    "box": 8.0,
    "cls": 0.41684,
    "dfl": 1.2,
    "hsv_h": 0.01469,
    "hsv_s": 0.6,
    "hsv_v": 0.44641,
    "optimizer": "AdamW",
}


# Generating the YAML file
def generate_yaml_file() -> None:
    config: dict = {
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
    client: PicselliaClient = PicselliaClient()

    dataset = client.get_dataset()
    download_dataset(INPUT_IMAGES_DIR, dataset)

    export_annotations(dataset, INPUT_ANNOTATIONS_DIR)
    extract_annotations(INPUT_ANNOTATIONS_DIR)

    experiment = client.get_experiment()

    # --- PART 2 : Split data for Ultralytics YOLO ---
    split_data_dict = get_split_data(INPUT_IMAGES_DIR, INPUT_ANNOTATIONS_DIR)

    # Copying files to corresponding directories
    for split, pairs in split_data_dict.items():
        copy_files(
            pairs,
            f"{DATASET_DIR}/{split}/{IMAGES_DIR}",
            f"{DATASET_DIR}/{split}/{LABELS_DIR}",
        )
    generate_yaml_file()

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(config.YOLO_MODEL)

    # Add callbacks for logs
    logger: PicselliaLogger = PicselliaLogger(client.get_experiment())
    model.add_callback("on_train_start", logger.on_train_start)
    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_train_end", logger.on_train_end)

    # Train the model using the dataset
    model.train(
        data=YAML_PATH,
        epochs=parameters["epochs"],
        seed=parameters["seed"],
        lr0=parameters["lr0"],
        lrf=parameters["lrf"],
        momentum=parameters["momentum"],
        weight_decay=parameters["weight_decay"],
        batch=parameters["batch"],
        patience=parameters["patience"],
        imgsz=parameters["imgsz"],
        plots=parameters["plots"],
        close_mosaic=parameters["close_mosaic"],
        box=parameters["box"],
        cls=parameters["cls"],
        dfl=parameters["dfl"],
        hsv_h=parameters["hsv_h"],
        hsv_s=parameters["hsv_s"],
        hsv_v=parameters["hsv_v"],
        optimizer=parameters["optimizer"],
        device=config.DEVICE
    )

    # Evaluate the model's performance on the validation set
    model.val()
    # print(results)

    # Store new model version
    model_name: str = "smarter" + time.strftime("-%Y-%m-%d-%H-%M-%S")
    model_version = client.create_model_version(model_name)

    model_version.store("model-latest", model.trainer.best, do_zip=True)

    experiment.log_parameters(parameters)
    experiment.attach_model_version(model_version, True)

    print("Experiment saved")


if __name__ == "__main__":
    main()
