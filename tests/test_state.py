from smart_pandas.state import State, StateName, MLStage


def test_state_from_data(smart_data):
    state = State.from_data(data=smart_data, config=smart_data.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.PROCESSED


def test_state_from_data_with_missing_target(smart_data):
    smart_data.drop(columns=["life_expectancy"], inplace=True)
    state = State.from_data(data=smart_data, config=smart_data.smart_pandas.config)
    assert state.ml_stage == MLStage.INFERENCE
    assert state.name == StateName.PROCESSED


def test_state_from_data_with_missing_model_features(smart_data):
    smart_data.drop(columns=["bmi"], inplace=True)
    state = State.from_data(data=smart_data, config=smart_data.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.RAW


def test_state_from_data_with_missing_raw_features(smart_data):
    smart_data.drop(columns=["height", "bmi"], inplace=True)
    state = State.from_data(data=smart_data, config=smart_data.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.UNKNOWN


def test_state_from_data_with_missing_unique_identifier(smart_data):
    smart_data.drop(columns=["user_id"], inplace=True)
    state = State.from_data(data=smart_data, config=smart_data.smart_pandas.config)
    assert state.ml_stage == MLStage.TRAINING
    assert state.name == StateName.CORRUPTED
