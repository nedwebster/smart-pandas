from enum import Enum

import pandas as pd

from smart_pandas.data_config import DataConfig


class StateName(Enum):
    RAW = "raw"
    PROCESSED = "processed"
    UNKNOWN = "unknown"
    CORRUPTED = "corrupted"


class MLStage(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class State:
    """
    Class for tracking the state of the data.

    The state is defined as a relationship between the columns in the data and the columns specified in the config.
    Different combinations of columns present in the data will result in different state settings.

    """
    def __init__(self, name: StateName, ml_stage: MLStage):
        """
        Initialise the state.

        Parameters
        ----------
        name: StateName
            The name of the state, representing specific steps in a processing pipeline, eg: raw, processed, etc.
        ml_stage: MLStage
            The ml stage of the state, representing the pipeline itself, eg: training or inference.
        """
        self.name = name
        self.ml_stage = ml_stage

    @classmethod
    def from_data(cls, data: pd.DataFrame, config: DataConfig) -> "State":
        """
        Attempt to infer the state of the data based on the config and the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to infer the task from.
        config: DataConfig
            The config to infer the task from.
        """
        state = cls(StateName.UNKNOWN, MLStage.INFERENCE)
        state.infer_task(data, config)
        state.infer_state(data, config)
        return state

    def infer_task(self, data: pd.DataFrame, config: DataConfig) -> None:
        """
        Attempt to infer the task based on the config and the data.

        Note: This method will return 'INFERENCE' for unsupervised tasks where a 'target' tag is not present in the
        config.

        Parameters
        ----------
        data: pd.DataFrame
            The data to infer the task from.
        config: DataConfig
            The config to infer the task from.
        """
        if config.target and all(x in data.columns for x in config.target):
            self.ml_stage = MLStage.TRAINING
        else:
            self.ml_stage = MLStage.INFERENCE

    def infer_state(self, data: pd.DataFrame, config: DataConfig) -> None:
        """
        Attempt to infer the state based on the config and the data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to infer the task from.
        config: DataConfig
            The config to infer the task from.
        """
        missing_column_counts = {
            tag: len([x for x in getattr(config, tag) if x not in data.columns])
            for tag in ["raw_features", "model_features", "target", "unique_identifier", "metadata", "row_timestamp"]
        }
        for tag in ["unique_identifier", "metadata", "row_timestamp"]:
            if missing_column_counts[tag] > 0:
                self.name = StateName.CORRUPTED
                return

        if self.ml_stage == MLStage.TRAINING:
            if missing_column_counts["target"] > 0:
                self.name = StateName.CORRUPTED
                return

        if missing_column_counts["model_features"] == 0:
            self.name = StateName.PROCESSED
            return

        if missing_column_counts["model_features"] > 0 and missing_column_counts["raw_features"] == 0:
            self.name = StateName.RAW
            return

        self.name = StateName.UNKNOWN

    def __str__(self):
        return f"{self.name.value} ({self.ml_stage.value})"

    def __repr__(self):
        return f"{self.name} ({self.ml_stage})"
