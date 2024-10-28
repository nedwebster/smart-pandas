def default_df_schema_params() -> dict:
    return {
        "coerce": True,
        "strict": False,  # want to allow columns to be missing if they're not created yet
        "unique_column_names": True,
        "add_missing_columns": False,  # make sure this is false to avoid adding calculated columns
    }
