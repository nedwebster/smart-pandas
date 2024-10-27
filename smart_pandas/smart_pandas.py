import pandas as pd
from smart_pandas.config_utils import read_config


@pd.api.extensions.register_dataframe_accessor("config")
class SmartPandas:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def init(self, config_path: str):
        self.config = read_config(config_path)

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
