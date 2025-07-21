import copy
import warnings
from typing import Any

import pandas as pd
from smart_pandas.config.config_utils import read_config
from smart_pandas.config.data_config import DataConfig
from smart_pandas.state import State, StateName, StateError, STATE_NAME_INCOMPATIBILITIES, ML_STAGE_INCOMPATIBILITIES
from smart_pandas.schema import build_schema
from smart_pandas.config.tag import TAGS

@pd.api.extensions.register_dataframe_accessor("smart_pandas")
class SmartPandas:
    """
    Pandas API extension for smart pandas functionality.
    
    This extension provides configuration-driven data validation and state management for machine learning workflows.
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
        self.config: DataConfig | None = None
        self.state: State | None = None
        self.schema: object | None = None
        self.auto_update: bool = True
        self.column_hash = hash(tuple(self._obj.columns))

    def load_config(
        self, 
        config_path: str | None = None, 
        config: DataConfig | None = None,
        auto_update: bool = True
    ) -> None:
        """
        Load the configuration for the SmartPandas accessor, and update the state and schema.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file
        config : DataConfig, optional
            DataConfig object to use directly
        auto_update : bool, default True
            Whether to automatically run update() after retrieving attributes

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
        self.auto_update = auto_update
        self.state = State.from_data(data=self._obj, config=self.config)
        self.schema = build_schema(self.config, self.state)
        self._set_data_attributes()

    def update(self) -> None:
        """Update SmartPandas properties if the datas column hash has changed."""
        if hash(tuple(self._obj.columns)) != self.column_hash:
            self.column_hash = hash(tuple(self._obj.columns))
            self._update_state()
            self._set_data_attributes()

    def _set_data_attributes(self) -> None:
        """Set the data attributes of the SmartPandas accessor based on the config and the current state."""
        self.name = self.config.name
        for tag in TAGS.values():
            if (
                tag.data_attribute_name not in STATE_NAME_INCOMPATIBILITIES[self.state.name]
                and tag.data_attribute_name not in ML_STAGE_INCOMPATIBILITIES[self.state.ml_stage]
            ):
                setattr(self, tag.data_attribute_name, self._obj[getattr(self.config, tag.data_attribute_name)])

    def _update_state(self) -> None:
        """
        Update the state of the DataFrame based on the config and current data.

        If the state has changed, the schema will be updated to reflect the new state.
        
        Raises
        ------
        RuntimeError
            If SmartPandas is not initialized
        """
        
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
    
    def validate(
        self, 
        inplace: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Validate the DataFrame using the Pandera schema based on the current state.

        Parameters
        ----------
        inplace : bool, default False
            Whether to validate the DataFrame in place or return a new validated DataFrame
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
        validated_data.smart_pandas.load_config(config=self.config)
        return validated_data

    def __getattribute__(self, name: str) -> Any:
        """Custom getter to allow validating and updating the data attributes when accessing them."""
        if name in [tag.data_attribute_name for tag in TAGS.values()] + ["state", "validate"]:
            if self.config is None:
                raise RuntimeError("SmartPandas not initialized. Call data.smart_pandas.load_config() first.")
            if self.auto_update:
                self.update()
        return super().__getattribute__(name)
