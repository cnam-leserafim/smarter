import argparse
import os
from enum import Enum

from dotenv import load_dotenv
from ultralytics import YOLO

from PicselliaClient import PicselliaClient

load_dotenv()


class InferenceMode(Enum):
    WEBCAM = 0
    IMAGE = 1
    VIDEO = 2


def get_model():
    client = PicselliaClient()
    project_model = client.get_model()
    model_versions = project_model.list_versions()
    latest_version = model_versions[-1]
    file = latest_version.get_file("best")
    file.download()
    return YOLO("best.pt")


def start_inference(mode: InferenceMode, path: str = None):
    model = get_model()

    if mode == InferenceMode.WEBCAM:
        model(0, device="cuda", show=True)
    elif mode in [InferenceMode.IMAGE, InferenceMode.VIDEO]:
        if not os.path.exists(path):
            print(f"Erreur: Le fichier {path} n'existe pas.")
            return
        result = model(path, device="cuda", show=True)
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
