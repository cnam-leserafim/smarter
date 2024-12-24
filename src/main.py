import os
import json
from dotenv import load_dotenv
from picsellia import Client

load_dotenv()

# Configuration du client
organization_name = "Picsalex-MLOps"
dataset_id = "0193688e-aa8f-7cbe-9396-bec740a262d0"

client = Client(
    api_token=os.getenv("PICSELLIA_API_TOKEN"), organization_name=organization_name
)
dataset = client.get_dataset_version_by_id(dataset_id)

output_dir_dataset = "./datasets"
os.makedirs(output_dir_dataset, exist_ok=True)
dataset.list_assets().download(output_dir_dataset)

project = client.get_project(project_name="Groupe_1")

experiment = project.create_experiment(
    name="experiment-0",
    description="base experiment",
)