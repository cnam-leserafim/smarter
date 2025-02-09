from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics.models.yolo.detect import DetectionTrainer


class PicselliaLogger:

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def on_train_start(self, trainer: DetectionTrainer):
        self.experiment.log("Model", str(trainer.model), LogType.LINE)

    def on_train_epoch_end(self, trainer: DetectionTrainer):
        for name, value in trainer.metrics.items():
            self.experiment.log(name, [value], LogType.LINE)
        self.experiment.log(
            "Epoch duration", [trainer.epoch_time], LogType.BAR
        )
        self.experiment.log("Fitness", [trainer.fitness], LogType.LINE)
        if trainer.best_fitness is not None:
            self.experiment.log(
                "Best fitness", trainer.best_fitness, LogType.VALUE
            )
        for index, loss_name in enumerate(trainer.loss_names):
            self.experiment.log(
                loss_name, [trainer.loss_items[index].item()], LogType.LINE
            )

    def on_train_end(self, trainer: DetectionTrainer):
        self.experiment.log("Model", str(trainer.model), LogType.LINE)
