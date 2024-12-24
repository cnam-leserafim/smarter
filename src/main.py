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