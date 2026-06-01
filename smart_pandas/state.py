from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Union

import pandas as pd

from smart_pandas.config.tag import TAGS

if TYPE_CHECKING:
    from smart_pandas.config.data_config import DataConfig

# Data types that the state engine can work with
DataLike = Union[pd.DataFrame, dict]


def _get_column_names(data: DataLike) -> Sequence[str]:
    """Get column/key names from a DataFrame or dict."""
    if isinstance(data, dict):
        return data.keys()
    return data.columns


class StateName(Enum):
    RAW = "raw"
    PROCESSED = "processed"
    UNKNOWN = "unknown"
    CORRUPTED = "corrupted"
    
    @property
    def incompatibilities(self) -> list[str]:
        return {
            StateName.RAW: ["model_features", "derived_features"],
            StateName.PROCESSED: [],
            StateName.UNKNOWN: ["raw_features", "derived_features", "model_features"],
            StateName.CORRUPTED: [tag.data_attribute_name for tag in TAGS.values()],
        }[self]


class MLStage(Enum):
    """Enum representing the ML pipeline stage."""
    
    # Format: (value, incompatible_attributes)
    TRAINING = "training"
    INFERENCE = "inference"

    @property
    def incompatibilities(self) -> list[str]:
        return {
            MLStage.TRAINING: [],
            MLStage.INFERENCE: ["target"],
        }[self]
    

class StateInferenceEngine:
    """Engine for inferring state from data and configuration."""
    
    @staticmethod
    def infer_ml_stage(data: DataLike, config: "DataConfig") -> MLStage:
        """
        Infer the ML stage based on presence of target columns.

        Parameters
        ----------
        data : DataLike
            The data to analyze (DataFrame or dict)
        config : DataConfig
            The configuration object

        Returns
        -------
        MLStage
            The inferred ML stage
        """
        columns = _get_column_names(data)
        if config.target and all(col in columns for col in config.target):
            return MLStage.TRAINING
        return MLStage.INFERENCE

    @staticmethod
    def infer_state_name(data: DataLike, config: "DataConfig", ml_stage: MLStage) -> StateName:
        """
        Infer the state name based on available columns.

        Parameters
        ----------
        data : DataLike
            The data to analyze (DataFrame or dict)
        config : DataConfig
            The configuration object
        ml_stage : MLStage
            The current ML stage

        Returns
        -------
        StateName
            The inferred state name
        """
        columns = _get_column_names(data)

        # Find missing columns in all tags
        missing_counts = {}
        for tag in TAGS.values():
            tag_cols = getattr(config, tag.data_attribute_name)
            missing_counts[tag.data_attribute_name] = sum(1 for col in tag_cols if col not in columns)

            if tag.required and missing_counts[tag.data_attribute_name] > 0:
                return StateName.CORRUPTED

        if ml_stage == MLStage.TRAINING and missing_counts["target"] > 0:
            return StateName.CORRUPTED

        # Determine processing state based on feature columns
        if missing_counts["model_features"] == 0:
            return StateName.PROCESSED
        elif missing_counts["derived_features"] == len(config.derived_features) and missing_counts["raw_features"] == 0:
            return StateName.RAW

        # return unknown if there are model features are partially computed
        return StateName.UNKNOWN

    @staticmethod
    def get_state_columns(config: "DataConfig", state: "State") -> list[str]:
        """
        Get the columns that are relevant to the current state.
        """

        columns = []
        data_attributes = [
            tag.data_attribute_name for tag in TAGS.values()
            if tag.data_attribute_name not in state.name.incompatibilities
            and tag.data_attribute_name not in state.ml_stage.incompatibilities
        ]
        for data_attribute in data_attributes:
            columns.extend(getattr(config, data_attribute))

        return columns

class State:
    """
    Class for tracking the state of ML data.

    The state represents the relationship between columns in the data and the expected
    configuration. It consists of a StateName (processing stage) and MLStage (pipeline type).
    """
    
    def __init__(self, name: StateName, ml_stage: MLStage):
        """
        Initialize the state.

        Parameters
        ----------
        name : StateName
            The processing stage of the data
        ml_stage : MLStage
            The ML pipeline stage (training or inference)
        """
        self.name = name
        self.ml_stage = ml_stage
        self._inference_engine = StateInferenceEngine()

    @classmethod
    def from_data(
        cls,
        data: DataLike,
        config: "DataConfig",
        state_name: "StateName | None" = None,
        ml_stage: "MLStage | None" = None,
    ) -> "State":
        """
        Create a State instance by inferring from data and configuration.

        Parameters
        ----------
        data : DataLike
            The data to analyze (DataFrame or dict)
        config : DataConfig
            The configuration object
        state_name : StateName, optional
            Override the inferred state name. When provided, skips state
            name inference.
        ml_stage : MLStage, optional
            Override the inferred ML stage. When provided, skips ML stage
            inference.

        Returns
        -------
        State
            The inferred (or overridden) state
        """
        inference_engine = StateInferenceEngine()
        ml_stage = ml_stage or inference_engine.infer_ml_stage(data, config)
        state_name = state_name or inference_engine.infer_state_name(data, config, ml_stage)
        return cls(state_name, ml_stage)

    def infer_state(self, data: DataLike, config: "DataConfig") -> None:
        """
        Update the state based on current data and configuration.

        Parameters
        ----------
        data : DataLike
            The data to analyze (DataFrame or dict)
        config : DataConfig
            The configuration object
        """
        self.name = self._inference_engine.infer_state_name(data, config, self.ml_stage)

    def get_state_columns(self, config: "DataConfig") -> list[str]:
        return StateInferenceEngine.get_state_columns(config, self)

    def __str__(self) -> str:
        return f"{self.name.value}, {self.ml_stage.value}"

    def __repr__(self) -> str:
        return f"State(name={self.name}, ml_stage={self.ml_stage})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, State):
            return False
        return self.name == other.name and self.ml_stage == other.ml_stage


class StateError(Exception):
    """Base class for state-related errors."""