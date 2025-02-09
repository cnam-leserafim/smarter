import argparse
import os
import glob
import shutil
from enum import Enum

from dotenv import load_dotenv
from ultralytics import YOLO

from picsellia_resources.picsellia_client import PicselliaClient

import zipfile

load_dotenv()

MODEL_NAME="model-latest"
MODEL_DIR_PATH = "./model"
DEVICE="cuda"

class InferenceMode(Enum):
    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


def get_model():
    client = PicselliaClient()
    latest_version = client.get_latest_version()
    model_file = latest_version.get_file(MODEL_NAME)
    model_file.download(MODEL_DIR_PATH)
    model_filename = os.path.join(MODEL_DIR_PATH, model_file.filename)
    with zipfile.ZipFile(model_filename, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR_PATH)
    pt_files = glob.glob(f"{MODEL_DIR_PATH}/**/*.pt", recursive=True)
    yolo_model = YOLO(pt_files[0])
    # delete all dirs and files in MODEL_DIR_PATH
    for item in os.listdir(MODEL_DIR_PATH):
        if os.path.isfile(os.path.join(MODEL_DIR_PATH, item)):
            os.remove(os.path.join(MODEL_DIR_PATH, item))
        else:
            shutil.rmtree(os.path.join(MODEL_DIR_PATH, item))
    return yolo_model


def start_inference(mode: InferenceMode, source_path: str = None) -> None:
    inference_model = get_model()

    if mode == InferenceMode.WEBCAM:
        inference_model(0, device=DEVICE, show=True)
    elif mode in [InferenceMode.IMAGE, InferenceMode.VIDEO]:
        if not os.path.exists(source_path):
            print(f"Erreur: Le fichier {source_path} n'existe pas.")
            return
        results = inference_model(source_path, device=DEVICE, show=True)
        if mode == InferenceMode.IMAGE:
            for result in results:
                result.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script d'inférence avec YOLO"
    )
    parser.add_argument(
        "--mode",
        choices=["WEBCAM", "IMAGE", "VIDEO"],
        required=True,
        help="Mode d'inférence",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Chemin du fichier (requis pour image et vidéo)",
    )
    args = parser.parse_args()

    if args.mode == "WEBCAM":
        start_inference(InferenceMode.WEBCAM)
    elif args.mode == "IMAGE":
        if args.path:
            start_inference(InferenceMode.IMAGE, args.path)
        else:
            print("Erreur: --path est requis pour l'inférence sur une image.")
    elif args.mode == "VIDEO":
        if args.path:
            start_inference(InferenceMode.VIDEO, args.path)
        else:
            print("Erreur: --path est requis pour l'inférence sur une vidéo.")
