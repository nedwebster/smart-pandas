import copy
import warnings
from typing import Optional

import pandas as pd
from smart_pandas.config.config_utils import read_config
from smart_pandas.config.data_config import DataConfig
from smart_pandas.state import State, StateName, StateError
from smart_pandas.schema import build_schema


@pd.api.extensions.register_dataframe_accessor("smart_pandas")
class SmartPandas:
    """
    Pandas API extension for smart pandas functionality.
    
    This extension provides configuration-driven data validation and state management
    for machine learning workflows.
    """
    
    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialize the SmartPandas accessor.
        
        Parameters
        ----------
        pandas_obj : pd.DataFrame
            The pandas DataFrame this accessor is attached to
        """
        self._obj = pandas_obj
        self.config: Optional[DataConfig] = None
        self.state: Optional[State] = None
        self.schema: Optional[object] = None

    def init(
        self, 
        config_path: Optional[str] = None, 
        config: Optional[DataConfig] = None
    ) -> None:
        """
        Initialize the configuration for the DataFrame.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file
        config : DataConfig, optional
            DataConfig object to use directly
            
        Raises
        ------
        ValueError
            If neither config nor config_path is provided
        TypeError
            If config is not a DataConfig instance
        """
        if config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided")
        
        if config is None:
            config = read_config(config_path)
            
        self.config = config
        self.state = State.from_data(data=self._obj, config=self.config)
        self.schema = build_schema(self.config, self.state)

    def _ensure_initialized(self) -> None:
        """Ensure the accessor has been initialized."""
        if self.config is None:
            raise RuntimeError("SmartPandas not initialized. Call init() first.")

    @property
    def name(self) -> str:
        """Get the dataset name from configuration."""
        self._ensure_initialized()
        return self.config.name

    @property
    def raw_features(self) -> pd.DataFrame:
        """Get raw feature columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.raw_features]

    @property
    def derived_features(self) -> pd.DataFrame:
        """Get derived feature columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.derived_features]

    @property
    def model_features(self) -> pd.DataFrame:
        """Get model feature columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.model_features]

    @property
    def target(self) -> pd.DataFrame:
        """Get target columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.target]

    @property
    def unique_identifier(self) -> pd.DataFrame:
        """Get unique identifier columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.unique_identifier]

    @property
    def metadata(self) -> pd.DataFrame:
        """Get metadata columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.metadata]

    @property
    def row_timestamp(self) -> pd.DataFrame:
        """Get row timestamp columns as a DataFrame."""
        self._ensure_initialized()
        return self._obj[self.config.row_timestamp]

    def validate(
        self, 
        inplace: bool = False, 
        update_state: bool = True, 
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Validate the DataFrame using the Pandera schema based on the current state.

        Parameters
        ----------
        inplace : bool, default False
            Whether to validate the DataFrame in place or return a new validated DataFrame
        update_state : bool, default True
            Whether to update the state of the DataFrame prior to validation
        **kwargs
            Additional keyword arguments to pass to the Pandera schema validate method

        Returns
        -------
        pd.DataFrame or None
            The validated DataFrame if inplace=False, None otherwise
            
        Raises
        ------
        RuntimeError
            If SmartPandas is not initialized
        ValueError
            If the data is in an invalid state for validation
        """
        self._ensure_initialized()
        
        if update_state:
            self.update_state()

        if self.state.name in [StateName.UNKNOWN, StateName.CORRUPTED]:
            raise StateError(
                f"Cannot validate data in {self.state.name.value} state. "
                "Please check your data and try again."
            )

        if inplace:
            self.schema.validate(self._obj, inplace=inplace, **kwargs)
            return None
            
        validated_data = self.schema.validate(self._obj, inplace=inplace, **kwargs)
        
        # Re-initialize the config after overwriting the DataFrame object pointer
        validated_data.smart_pandas.init(config=self.config)
        return validated_data

    def update_state(self) -> None:
        """
        Update the state of the DataFrame based on the config and current data.

        If the state has changed, the schema will be updated to reflect the new state.
        
        Raises
        ------
        RuntimeError
            If SmartPandas is not initialized
        """
        self._ensure_initialized()
        
        old_state = copy.deepcopy(self.state)
        self.state.infer_state(data=self._obj, config=self.config)

        if self.state.name in [StateName.CORRUPTED, StateName.UNKNOWN]:
            warnings.warn(
                f"The state of the DataFrame is {self.state.name.value}. "
                "Please check your data and try again.",
                UserWarning,
                stacklevel=2
            )

        if old_state != self.state:
            self.schema = build_schema(self.config, self.state)
