def default_df_schema_params() -> dict:
    return {
        "coerce": True,
        "unique_column_names": True,
        "add_missing_columns": False,  # make sure this is false to avoid adding calculated columns
    }
