import argparse
import os
from enum import Enum

from dotenv import load_dotenv
from ultralytics import YOLO

from picsellia_resources.picsellia_client import PicselliaClient

load_dotenv()


class InferenceMode(Enum):
    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


def get_model():
    client = PicselliaClient()
    latest_version = client.get_latest_version()
    file = latest_version.get_file("best")
    file.download()
    return YOLO("best.pt")


def start_inference(mode: InferenceMode, source_path: str = None) -> None:
    inference_model = get_model()

    if mode == InferenceMode.WEBCAM:
        inference_model(0, device="cuda", show=True)
    elif mode in [InferenceMode.IMAGE, InferenceMode.VIDEO]:
        if not os.path.exists(source_path):
            print(f"Erreur: Le fichier {source_path} n'existe pas.")
            return
        result = inference_model(source_path, device="cuda", show=True)
        if mode == InferenceMode.IMAGE:
            result[0].show()


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
