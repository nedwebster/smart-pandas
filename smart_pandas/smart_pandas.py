from typing import Optional

import pandas as pd
from smart_pandas.column_set import ColumnSet
from smart_pandas.config_utils import read_config


@pd.api.extensions.register_dataframe_accessor("config")
class SmartPandas:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def init(
        self, config_path: Optional[str] = None, config: Optional[ColumnSet] = None
    ):
        if config is None:
            if config_path is None:
                raise ValueError("Either config or config_path must be provided")
            config = read_config(config_path)
        self.config = config

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
    def schema(self):
        return self.config.schema

    def validate(self):
        validated_data = self.config.schema.validate(self._obj)
        validated_data.config.init(config=self.config)
        return validated_data
