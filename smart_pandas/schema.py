
import pandera as pa
from smart_pandas.config.data_config import DataConfig
from smart_pandas.state import State


def get_default_df_schema_params() -> dict:
    return {
        "coerce": True,
        "unique_column_names": True,
        "add_missing_columns": False,  # make sure this is false to avoid adding calculated columns
        "strict": True,
    }


def build_schema(config: DataConfig, state: State) -> pa.DataFrameSchema:
    """
    Build a schema for the dataframe based on the config and the state.

    Parameters
    ----------
    config: DataConfig
        The config to build the schema from.
    state: State
        The state to build the schema from.
    """
    state_columns = state.get_state_columns(config)
    column_schemas = {column.name: column.data_schema for column in config.columns if column.name in state_columns}
    return pa.DataFrameSchema(column_schemas, **get_default_df_schema_params())
