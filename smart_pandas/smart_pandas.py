import copy
import warnings
from typing import Optional

import pandas as pd
from smart_pandas.config.config_utils import read_config
from smart_pandas.config.data_config import DataConfig
from smart_pandas.state import State, StateName
from smart_pandas.schema import build_schema


@pd.api.extensions.register_dataframe_accessor("smart_pandas")
class SmartPandas:
    """Pandas API extension for smart pandas."""
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def init(
        self, config_path: Optional[str] = None, config: Optional[DataConfig] = None
    ):
        """Initialise the config for the dataframe via either a config path or a config object."""
        if config is None:
            if config_path is None:
                raise ValueError("Either config or config_path must be provided")
            config = read_config(config_path)
        self.config = config
        self.state = State.from_data(data=self._obj, config=self.config)
        self.schema = build_schema(self.config, self.state)

    @property
    def name(self):
        return self.config.name

    @property
    def raw_features(self):
        return self._obj[self.config.raw_features]

    @property
    def derived_features(self):
        return self._obj[self.config.derived_features]

    @property
    def model_features(self):
        return self._obj[self.config.model_features]

    @property
    def target(self):
        return self._obj[self.config.target]

    @property
    def unique_identifier(self):
        return self._obj[self.config.unique_identifier]

    @property
    def metadata(self):
        return self._obj[self.config.metadata]

    @property
    def row_timestamp(self):
        return self._obj[self.config.row_timestamp]

    def validate(self, inplace: bool = False, update_state: bool = True, **kwargs):
        """
        Validate the DataFrame using the pandera schema, based on the current state.

        Parameters
        ----------
        inplace : bool
            Whether to validate the DataFrame in place or return a new validated DataFrame.
        update_state : bool
            Whether to update the state of the dataframe prior to validation.
        kwargs
            Additional keyword arguments to pass to the pandera schema validate method.

        Returns
        -------
        validated_data: pd.DataFrame or None
            The validated DataFrame. If inplace is True, returns None.
        """
        if update_state:
            self.update_state()

        if self.state in [StateName.UNKNOWN, StateName.CORRUPTED]:
            raise ValueError(
                f"Cannot validate data in {self.state.name.value} state, please check your data and try again."
            )

        if inplace:
            self.schema.validate(self._obj, inplace=inplace, **kwargs)
            return None
        validated_data = self.schema.validate(
            self._obj, inplace=inplace, **kwargs
        )

        # re-initialise the config after overwriting the dataframe object pointer
        validated_data.smart_pandas.init(config=self.config)
        return validated_data

    def update_state(self):
        """
        Update the state of the dataframe based on the config and the data.

        If the state has changed, the schema will be updated to reflect the new state.
        """
        old_state = copy.deepcopy(self.state)
        self.state.infer_state(data=self._obj, config=self.config)

        if self.state.name in [StateName.CORRUPTED, StateName.UNKNOWN]:
            warnings.warn(
                "The state of the dataframe is either corrupt or unknown, please check your data and try again.",
                UserWarning,
            )

        if old_state != self.state:
            self.schema = build_schema(self.config, self.state)
