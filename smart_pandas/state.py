from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from smart_pandas.config.data_config import DataConfig


class StateName(Enum):
    RAW = "raw"
    PROCESSED = "processed"
    UNKNOWN = "unknown"
    CORRUPTED = "corrupted"


class MLStage(Enum):
    TRAINING = "training"
    INFERENCE = "inference"


class StateInferenceEngine:
    """Engine for inferring state from data and configuration."""
    
    @staticmethod
    def infer_ml_stage(data: pd.DataFrame, config: "DataConfig") -> MLStage:
        """
        Infer the ML stage based on presence of target columns.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to analyze
        config : DataConfig
            The configuration object
            
        Returns
        -------
        MLStage
            The inferred ML stage
        """
        if config.target and all(col in data.columns for col in config.target):
            return MLStage.TRAINING
        return MLStage.INFERENCE
    
    @staticmethod
    def infer_state_name(data: pd.DataFrame, config: "DataConfig", ml_stage: MLStage) -> StateName:
        """
        Infer the state name based on available columns.
        
        Parameters
        ----------
        data : pd.DataFrame
            The data to analyze
        config : DataConfig
            The configuration object
        ml_stage : MLStage
            The current ML stage
            
        Returns
        -------
        StateName
            The inferred state name
        """
        # Check for missing columns in critical categories
        missing_counts = {}
        for tag in ["raw_features", "model_features", "target", "unique_identifier", "metadata", "row_timestamp"]:
            required_cols = getattr(config, tag)
            missing_counts[tag] = sum(1 for col in required_cols if col not in data.columns)
        
        # Check for corrupted state - missing required columns
        for tag in ["unique_identifier", "metadata", "row_timestamp"]:
            if missing_counts[tag] > 0:
                return StateName.CORRUPTED
        
        # Check target columns for training stage
        if ml_stage == MLStage.TRAINING and missing_counts["target"] > 0:
            return StateName.CORRUPTED
        
        # Determine processing state
        if missing_counts["model_features"] == 0:
            return StateName.PROCESSED
        elif missing_counts["model_features"] > 0 and missing_counts["raw_features"] == 0:
            return StateName.RAW
        
        return StateName.UNKNOWN


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
    def from_data(cls, data: pd.DataFrame, config: "DataConfig") -> "State":
        """
        Create a State instance by inferring from data and configuration.

        Parameters
        ----------
        data : pd.DataFrame
            The data to analyze
        config : DataConfig
            The configuration object
            
        Returns
        -------
        State
            The inferred state
        """
        inference_engine = StateInferenceEngine()
        ml_stage = inference_engine.infer_ml_stage(data, config)
        state_name = inference_engine.infer_state_name(data, config, ml_stage)
        return cls(state_name, ml_stage)

    def infer_state(self, data: pd.DataFrame, config: "DataConfig") -> None:
        """
        Update the state based on current data and configuration.

        Parameters
        ----------
        data : pd.DataFrame
            The data to analyze
        config : DataConfig
            The configuration object
        """
        self.name = self._inference_engine.infer_state_name(data, config, self.ml_stage)

    def get_state_columns(self, config: "DataConfig") -> list[str]:
        """
        Get the columns that are relevant to the current state.

        Parameters
        ----------
        config : DataConfig
            The configuration object
            
        Returns
        -------
        list[str]
            List of column names relevant to the current state
        """
        columns = config.unique_identifier + config.metadata + config.row_timestamp
        
        if self.ml_stage == MLStage.TRAINING:
            columns.extend(config.target)
            
        if self.name == StateName.PROCESSED:
            columns.extend(config.model_features)
        elif self.name == StateName.RAW:
            columns.extend(config.raw_features)
            
        return columns

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