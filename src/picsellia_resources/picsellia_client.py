import os

from picsellia import Client, DatasetVersion, Experiment, Model, ModelVersion
from picsellia.types.enums import Framework, InferenceType

import src.config as config

class PicselliaClient:

    def __init__(self):
        try:
            self.__client = Client(
                api_token=os.getenv("PICSELLIA_API_TOKEN"),
                organization_name=config.WORKSPACE_NAME,
            )
            self.__project = self.__client.get_project(
                project_name=config.PROJECT_NAME
            )
        except Exception as e:
            print("Unable to initialize Picsellia client : ", e)

    def get_dataset(self) -> DatasetVersion:
        try:
            return self.__client.get_dataset_version_by_id(config.DATASET_ID)
        except Exception as e:
            print("Unable to fetch dataset : ", e)

    def get_experiment(self) -> Experiment:
        try:
            experiment: Experiment = self.__project.get_experiment(
                name=config.EXPERIMENT_NAME
            )
            print(f"Existing experimentation recovered : {experiment.name}")
        except Exception as e:
            print("Unable to fetch experiment : ", e)
            experiment = self.__project.create_experiment(
                name=config.EXPERIMENT_NAME,
                description="base experiment",
            )

            experiment.attach_dataset(
                name=config.DATASET_NAME,
                dataset_version=self.get_dataset(),
            )
            print(f"Creation of new experiment : {experiment.name}")
        return experiment

    def get_model(self) -> Model:
        return self.__client.get_model(config.MODEL_NAME)

    def create_model_version(self, name: str) -> ModelVersion:
        model: Model = self.get_model()
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

    def get_latest_version(self) -> ModelVersion:
        model: Model = self.get_model()
        versions_list = model.list_versions(order_by=["-version"])
        return versions_list[0]
