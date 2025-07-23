
from typing import TYPE_CHECKING

import pandera as pa

if TYPE_CHECKING:
    from smart_pandas.config.data_config import DataConfig
    from smart_pandas.state import State


def get_default_df_schema_params() -> dict[str, bool]:
    """
    Get default parameters for DataFrame schema validation.
    
    Returns
    -------
    dict[str, bool]
        Dictionary of default schema parameters
    """
    return {
        "coerce": True,
        "unique_column_names": True,
        "add_missing_columns": False,  # Ensure we don't add calculated columns
        "strict": True,
    }


def build_schema(config: "DataConfig", state: "State") -> pa.DataFrameSchema:
    """
    Build a Pandera schema for the DataFrame based on configuration and state.

    Parameters
    ----------
    config : DataConfig
        The configuration object containing column definitions
    state : State
        The current state of the data (determines which columns to include)
        
    Returns
    -------
    pa.DataFrameSchema
        Configured Pandera schema for validation
        
    Raises
    ------
    ValueError
        If no columns are found for the current state
    """
    state_columns = state.get_state_columns(config)
    
    if not state_columns:
        raise ValueError(f"No columns found for state: {state}")
    
    column_schemas = {
        column.name: column.data_schema 
        for column in config.columns 
        if column.name in state_columns
    }
    
    if not column_schemas:
        raise ValueError(f"No valid column schemas found for state: {state}")
    
    return pa.DataFrameSchema(column_schemas, **get_default_df_schema_params())
