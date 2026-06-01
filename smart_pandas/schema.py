
import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandera as pa
from pydantic import ConfigDict, create_model

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pyspark.sql.types import StructType

    from smart_pandas.config.data_config import DataConfig
    from smart_pandas.state import State


# Mapping from numpy dtype categories to Python types for Pydantic field annotations
_NUMPY_TO_PYTHON_TYPE: dict[type, type] = {
    np.integer: int,
    np.floating: float,
    np.str_: str,
    np.bool_: bool,
    np.datetime64: datetime.datetime,
}


def _pa_column_to_python_type(pa_column: pa.Column) -> type:
    """Convert a Pandera column dtype to a Python type for Pydantic."""
    numpy_type = pa_column.dtype.type.type
    for np_base, py_type in _NUMPY_TO_PYTHON_TYPE.items():
        if np.issubdtype(numpy_type, np_base):
            return py_type
    return Any


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
        column.name: column.get_data_schema(state.name)
        for column in config.columns
        if column.name in state_columns
    }

    if not column_schemas:
        raise ValueError(f"No valid column schemas found for state: {state}")

    return pa.DataFrameSchema(column_schemas, **get_default_df_schema_params())


def build_dict_schema(config: "DataConfig", state: "State") -> type["BaseModel"]:
    """
    Build a Pydantic model for dict validation based on configuration and state.

    Mirrors ``build_schema`` but produces a Pydantic model instead of a
    Pandera ``DataFrameSchema``, suitable for validating single-record dicts
    without converting to a DataFrame.

    Results are cached on the config object by (state name, ML stage) so that
    repeated calls with the same config and state reuse the model class
    rather than rebuilding it via ``create_model``.

    Parameters
    ----------
    config : DataConfig
        The configuration object containing column definitions
    state : State
        The current state of the data (determines which fields to include)

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class

    Raises
    ------
    ValueError
        If no columns are found for the current state
    """
    if not hasattr(config, "_dict_schema_cache"):
        config._dict_schema_cache = {}

    cache_key = (state.name, state.ml_stage)
    if cache_key in config._dict_schema_cache:
        return config._dict_schema_cache[cache_key]

    state_columns = state.get_state_columns(config)

    if not state_columns:
        raise ValueError(f"No columns found for state: {state}")

    field_definitions: dict[str, Any] = {}
    for column in config.columns:
        if column.name not in state_columns:
            continue

        pa_col = column.get_data_schema(state.name)
        py_type = _pa_column_to_python_type(pa_col)

        if pa_col.nullable:
            py_type = Optional[py_type]

        if pa_col.required:
            field_definitions[column.name] = (py_type, ...)
        else:
            field_definitions[column.name] = (py_type, None)

    model = create_model(
        f"DictSchema_{state.name.value}_{state.ml_stage.value}",
        __config__=ConfigDict(extra="forbid"),
        **field_definitions,
    )
    config._dict_schema_cache[cache_key] = model
    return model


def build_pyspark_schema(config: "DataConfig", state: "State") -> "StructType":
    """
    Build a PySpark StructType schema based on configuration and state.

    Mirrors ``build_schema`` but produces a PySpark schema suitable for
    creating Spark DataFrames or writing to Delta Tables.

    Parameters
    ----------
    config : DataConfig
        The configuration object containing column definitions
    state : State
        The current state of the data (determines which fields to include)

    Returns
    -------
    pyspark.sql.types.StructType
        A PySpark schema

    Raises
    ------
    ValueError
        If no columns are found for the current state
    ImportError
        If pyspark is not installed
    """
    try:
        from pyspark.sql.types import (
            BooleanType,
            DoubleType,
            LongType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )
    except ImportError:
        raise ImportError(
            "pyspark is required for build_pyspark_schema. "
            "Install it with: pip install smart-pandas[pyspark]"
        )

    numpy_to_spark = {
        np.integer: LongType(),
        np.floating: DoubleType(),
        np.str_: StringType(),
        np.bool_: BooleanType(),
        np.datetime64: TimestampType(),
    }

    def _pa_column_to_spark_type(pa_column: pa.Column):
        numpy_type = pa_column.dtype.type.type
        for np_base, spark_type in numpy_to_spark.items():
            if np.issubdtype(numpy_type, np_base):
                return spark_type
        return StringType()

    state_columns = state.get_state_columns(config)

    if not state_columns:
        raise ValueError(f"No columns found for state: {state}")

    fields = []
    for column in config.columns:
        if column.name not in state_columns:
            continue

        pa_col = column.get_data_schema(state.name)
        spark_type = _pa_column_to_spark_type(pa_col)
        nullable = pa_col.nullable

        fields.append(StructField(column.name, spark_type, nullable=nullable))

    return StructType(fields)
