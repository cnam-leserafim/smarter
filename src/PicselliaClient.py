import os

from picsellia import Client, DatasetVersion, Experiment, Model, ModelVersion
from picsellia.types.enums import AnnotationFileType, Framework, InferenceType

# PICSELLIA
WORKSPACE_NAME = "Picsalex-MLOps"
DATASET_ID = "0193688e-aa8f-7cbe-9396-bec740a262d0"

PROJECT_ID = "0193641e-53c8-7928-887c-4be047938648"
PROJECT_NAME = "Groupe_1"

EXPERIMENT_ID = "01938deb-1008-755a-86b5-a24d3f9f6318"
EXPERIMENT_NAME = "experiment-0"

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
        project = self.client.get_project(project_name=PROJECT_NAME)
        return project.get_experiment(name=EXPERIMENT_NAME)

    def import_experiment(self):
        # Existing experiment
        project = self.client.get_project(project_name=PROJECT_NAME)
        experiment = project.get_experiment(name=EXPERIMENT_NAME)
        print(f"Existing experimentation recovered : {experiment.name}")
        datasets = experiment.list_attached_dataset_versions()
        print(f"Datasets attached to the experience : {datasets}")
        """
            experiment = project.create_experiment(
                name="experiment-1",
                description="base experiment",
            )

            experiment.attach_dataset(
                name="⭐️ cnam_product_2024",
                dataset_version=dataset,
            )
            print(f"Creation of new experiment : {experiment.name}")
            """

    # Export of annotations in YOLO format
    @staticmethod
    def export_annotations(dataset: DatasetVersion, input_dir: str):
        os.makedirs(input_dir, exist_ok=True)
        dataset.export_annotation_file(AnnotationFileType.YOLO, input_dir)
        print(f"Annotations exported to : {input_dir}")

    def get_model(self) -> Model:
        return self.client.get_model(MODEL_NAME)

    def create_model_version(self, name: str) -> ModelVersion:
        model = self.get_model()
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
