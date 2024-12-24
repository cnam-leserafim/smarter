import os
import json
import zipfile
from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import AnnotationFileType

load_dotenv()
WORKSPACE_NAME = "Picsalex-MLOps"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"
OUTPUT_DIR_DATASET = "./datasets"
ANNOTATIONS_DIR = f"{OUTPUT_DIR_DATASET}/annotations"


# Connexion au client Picsellia
def connect_to_client():
    return Client(
        api_token=os.getenv("PICSELLIA_API_TOKEN"),
        organization_name=WORKSPACE_NAME
    )
# Téléchargement du dataset depuis Picsellia
def import_datasets(client):
    dataset = client.get_dataset_version_by_id(DATASET_ID)
    os.makedirs(OUTPUT_DIR_DATASET, exist_ok=True)
    dataset.list_assets().download(OUTPUT_DIR_DATASET)
    print("Datasets importés")
    return dataset


def get_experiment(client):
    # Expérimentation existante
    project = client.get_project(project_name="Groupe_1")
    experiment = project.get_experiment(name="experiment-0")
    print(f"Expérimentation existante récupérée : {experiment.name}")
    datasets = experiment.list_attached_dataset_versions()
    print(f"Datasets attachés à l'expérience : {datasets}")
    '''
        experiment = project.create_experiment(
            name="experiment-1",
            description="base experiment",
        )

        experiment.attach_dataset(
            name="⭐️ cnam_product_2024",
            dataset_version=dataset,
        )
        print(f"Création nouvelle expérimentation : {experiment.name}")
        '''
# Export des annotations au format YOLO
def export_annotations(dataset):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    zip_path = os.path.join(ANNOTATIONS_DIR, "annotations.zip")
    dataset.export_annotation_file(AnnotationFileType.YOLO, zip_path)
    print(f"Annotations exportées dans : {ANNOTATIONS_DIR}")

def extract_annotations():
    # Trouver la première archive ZIP dans le dossier ou sous-dossier
    zip_file = next(
        (os.path.join(root, file) for root, _, files in os.walk(ANNOTATIONS_DIR)
         for file in files if file.endswith(".zip")),
        None,
    )
    if zip_file:
        # Créer le dossier "annotations" s'il n'existe pas
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

        # Décompresser l'archive ZIP
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(ANNOTATIONS_DIR)
        print(f"Archive décompressée dans : {ANNOTATIONS_DIR}")

        # Supprimer l'archive ZIP
        os.remove(zip_file)
        print(f"Archive {zip_file} supprimée.")
    else:
        print("Aucune archive ZIP trouvée dans le dossier 'datasets' ou ses sous-dossiers.")

    # Vérifier les fichiers extraits
    extracted_files = os.listdir(ANNOTATIONS_DIR)
    print(f"Fichiers extraits : {extracted_files}")
    file_count = len(extracted_files)
    print(f"Nombre total de fichiers extraits : {file_count}")

def main():
    # --- PARTIE 1 : Importer les images et les annotations ---
    client = connect_to_client()
    dataset = import_datasets(client)
    get_experiment(client)
    export_annotations(dataset)
    extract_annotations()


if __name__ == "__main__":
    main()