import os

from picsellia import Client, DatasetVersion, Experiment, ModelVersion
from picsellia.types.enums import AnnotationFileType, Framework, InferenceType

# PICSELLIA
WORKSPACE_NAME = "Picsalex-MLOps"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"

PROJECT_ID = "Groupe_1"

EXPERIMENT_ID = "experiment-4"

MODEL_ID = "019493d3-d97b-71a9-9051-3d558aedf5f4"
MODEL_NAME = "smarter"


class PicselliaClient:

    def __init__(self):
        self.client = Client(
            api_token=os.getenv("PICSELLIA_API_TOKEN"),
            organization_name=WORKSPACE_NAME,
        )

    def get_dataset(self) -> DatasetVersion:
        return self.client.get_dataset_version_by_id(DATASET_ID)

    # Downloading the dataset from Picsellia
    def import_datasets(self, input_dir: str) -> DatasetVersion:
        dataset = self.client.get_dataset_version_by_id(DATASET_ID)
        os.makedirs(input_dir, exist_ok=True)
        dataset.list_assets().download(input_dir)
        print("Imported datasets")
        return dataset

    def get_experiment(self) -> Experiment:
        project = self.client.get_project(project_name=PROJECT_ID)
        return project.get_experiment(name=EXPERIMENT_ID)

    def import_experiment(self):
        # Existing experiment
        project = self.client.get_project(project_name=PROJECT_ID)
        try:
            experiment = project.get_experiment(name=EXPERIMENT_ID)
            print(f"Existing experimentation recovered : {experiment.name}")
            datasets = experiment.list_attached_dataset_versions()
            print(f"Datasets attached to the experience : {datasets}")
        except Exception as e:
            experiment = project.create_experiment(
                name=EXPERIMENT_ID,
                description="base experiment",
            )

            experiment.attach_dataset(
                name="⭐️ cnam_product_2024",
                dataset_version=self.get_dataset(),
            )
            print(f"Creation of new experiment : {experiment.name}")

    # Export of annotations in YOLO format
    @staticmethod
    def export_annotations(dataset: DatasetVersion, input_dir: str):
        os.makedirs(input_dir, exist_ok=True)
        dataset.export_annotation_file(AnnotationFileType.YOLO, input_dir)
        print(f"Annotations exported to : {input_dir}")

    def create_model_version(self, name: str) -> ModelVersion:
        model = self.client.get_model(MODEL_NAME)
        labelmap: dict = {
            str(index): label.name
            for index, label in enumerate(self.get_dataset().list_labels())
        }
        return model.create_version(
            name=name,  # <- The name of your ModelVersion
            labels=labelmap,
            type=InferenceType.OBJECT_DETECTION,
            framework=Framework.ONNX,
        )
