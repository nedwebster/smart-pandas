from typing import Optional

import pandas as pd
from smart_pandas.column_set import ColumnSet
from smart_pandas.config_utils import read_config


@pd.api.extensions.register_dataframe_accessor("smart_pandas")
class SmartPandas:
    """Pandas API extension for smart pandas."""
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def init(
        self, config_path: Optional[str] = None, config: Optional[ColumnSet] = None
    ):
        """Initialise the config for the dataframe via either a config path or a config object."""
        if config is None:
            if config_path is None:
                raise ValueError("Either config or config_path must be provided")
            config = read_config(config_path)
        self.smart_pandas = config

    @property
    def name(self):
        return self.smart_pandas.name

    @property
    def raw_features(self):
        return self._obj[self.smart_pandas.raw_features]

    @property
    def derived_features(self):
        return self._obj[self.smart_pandas.derived_features]

    @property
    def model_features(self):
        return self._obj[self.smart_pandas.model_features]

    @property
    def target(self):
        return self._obj[self.smart_pandas.target]

    @property
    def unique_identifier(self):
        return self._obj[self.smart_pandas.unique_identifier]

    @property
    def metadata(self):
        return self._obj[self.smart_pandas.metadata]

    @property
    def row_timestamp(self):
        return self._obj[self.smart_pandas.row_timestamp]

    @property
    def schema(self):
        return self.smart_pandas.schema

    def validate(self, inplace: bool = False, **kwargs):
        """
        Validate the DataFrame using the pandera schema defined in the config.

        Parameters
        ----------
        inplace : bool
            Whether to validate the DataFrame in place or return a new validated DataFrame.
        kwargs
            Additional keyword arguments to pass to the pandera schema validate method.

        Returns
        -------
        validated_data: pd.DataFrame or None
            The validated DataFrame. If inplace is True, returns None.
        """
        if inplace:
            self.smart_pandas.schema.validate(self._obj, inplace=inplace, **kwargs)
            return None
        validated_data = self.smart_pandas.schema.validate(
            self._obj, inplace=inplace, **kwargs
        )
        validated_data.smart_pandas.init(config=self.smart_pandas)
        return validated_data
