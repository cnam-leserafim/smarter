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


# Connect to the Picsellia client
def connect_to_client():
    return Client(
        api_token=os.getenv("PICSELLIA_API_TOKEN"),
        organization_name=WORKSPACE_NAME
    )

# Downloading the dataset from Picsellia
def import_datasets(client):
    dataset = client.get_dataset_version_by_id(DATASET_ID)
    os.makedirs(OUTPUT_DIR_DATASET, exist_ok=True)
    dataset.list_assets().download(OUTPUT_DIR_DATASET)
    print("Imported datasets")
    return dataset


def get_experiment(client):
    # Existing experiment
    project = client.get_project(project_name="Groupe_1")
    experiment = project.get_experiment(name="experiment-0")
    print(f"Existing experimentation recovered : {experiment.name}")
    datasets = experiment.list_attached_dataset_versions()
    print(f"Datasets attached to the experience : {datasets}")
    '''
        experiment = project.create_experiment(
            name="experiment-1",
            description="base experiment",
        )

        experiment.attach_dataset(
            name="⭐️ cnam_product_2024",
            dataset_version=dataset,
        )
        print(f"Creation of new experiment : {experiment.name}")
        '''

# Export of annotations in YOLO format
def export_annotations(dataset):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    zip_path = os.path.join(ANNOTATIONS_DIR, "annotations.zip")
    dataset.export_annotation_file(AnnotationFileType.YOLO, zip_path)
    print(f"Annotations exported to : {ANNOTATIONS_DIR}")

def extract_annotations():
    # Find the first ZIP archive in the folder or subfolder
    zip_file = next(
        (os.path.join(root, file) for root, _, files in os.walk(ANNOTATIONS_DIR)
         for file in files if file.endswith(".zip")),
        None,
    )
    if zip_file:
        # Create the "annotations" folder if it does not exist
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

        # Unzip the ZIP archive
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(ANNOTATIONS_DIR)
        print(f"Archive unzipped in : {ANNOTATIONS_DIR}")

        # Delete ZIP archive
        os.remove(zip_file)
        print(f"Archive {zip_file} deleted.")
    else:
        print("No ZIP archive found in 'datasets' folder or its subfolders.")

    # Check extracted files
    extracted_files = os.listdir(ANNOTATIONS_DIR)
    print(f"Extracted files: {extracted_files}")
    file_count = len(extracted_files)
    print(f"Total number of extracted files : {file_count}")

def main():
    # --- PART 1: Import images and annotations ---
    client = connect_to_client()
    dataset = import_datasets(client)
    get_experiment(client)
    export_annotations(dataset)
    extract_annotations()


if __name__ == "__main__":
    main()