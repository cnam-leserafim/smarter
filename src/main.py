import os
import json
from dotenv import load_dotenv
from picsellia import Client

load_dotenv()
WORKSPACE_NAME = "Picsalex-MLOps"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"
OUTPUT_DIR_DATASET = "./datasets"

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

def main():
    # --- PARTIE 1 : Importer les images et les annotations ---
    client = connect_to_client()
    dataset = import_datasets(client)
    get_experiment(client)


if __name__ == "__main__":
    main()