from smart_pandas.state import State, StateName, MLStage


def test_state_from_data(smart_data_raw):
    state = State.from_data(data=smart_data_raw, config=smart_data_raw.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.RAW


def test_state_from_data_with_missing_target(smart_data_processed):
    smart_data_processed.drop(columns=["life_expectancy"], inplace=True)
    state = State.from_data(data=smart_data_processed, config=smart_data_processed.smart_pandas.config)
    assert state.ml_stage == MLStage.INFERENCE
    assert state.name == StateName.PROCESSED


def test_state_from_data_with_missing_model_features(smart_data_processed):
    state = State.from_data(data=smart_data_processed, config=smart_data_processed.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.PROCESSED


def test_state_from_data_with_missing_raw_features(smart_data_processed):
    smart_data_processed.drop(columns=["height", "bmi"], inplace=True)
    state = State.from_data(data=smart_data_processed, config=smart_data_processed.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.UNKNOWN


def test_state_from_data_with_missing_unique_identifier(smart_data_processed):
    smart_data_processed.drop(columns=["user_id"], inplace=True)
    state = State.from_data(data=smart_data_processed, config=smart_data_processed.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.CORRUPTED
