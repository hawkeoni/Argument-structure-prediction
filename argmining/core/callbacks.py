import re
import os
from typing import Dict, Any
from pathlib import Path

import wandb
from allennlp.training import EpochCallback, GradientDescentTrainer


@EpochCallback.register("wandb")
class WandbCallback(EpochCallback):
    """
    Callback to log metrics to wandb.
    """

    def __init__(
        self,
        project: str,
        run_name: str = None,
        metrics_regex: str = ".*",
    ):
        self.project = project
        self.metrics_regex = metrics_regex
        self.run = wandb.init(project=project, job_type="train")
        self.saved_config = False
        if run_name is not None:
            print("Run name deprecated")

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ):
        if not is_master:
            return
        if not self.saved_config:
            self.saved_config = True
            artifact = wandb.Artifact(name=trainer._serialization_dir, type="config")
            config_path = str(Path(trainer._serialization_dir) / "config.json")
            artifact.add_file(config_path)
            self.run.log_artifact(artifact)
            self.run.name = trainer._serialization_dir

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (float, int)) and re.findall(
                self.metrics_regex, metric_name
            ):
                wandb.log({"epoch": epoch, metric_name: metric_value})
