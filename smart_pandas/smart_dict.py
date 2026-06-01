from typing import Any

from pydantic import ValidationError

from smart_pandas.config.config_utils import read_config
from smart_pandas.config.data_config import DataConfig
from smart_pandas.config.tag import TAGS
from smart_pandas.schema import build_dict_schema, build_pyspark_schema
from smart_pandas.state import State, StateName, StateError


_STATE_ATTRS = frozenset(
    [tag.data_attribute_name for tag in TAGS.values()] + ["state", "validate"]
)


class DictValidationError(Exception):
    """Raised when dict validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class SmartDict(dict):
    """A config-aware dict for single-record validation.

    Subclasses dict so all standard dict operations work natively (iteration,
    .keys(), .values(), .items(), len(), etc.). State is lazily re-inferred
    only when ``state``, ``validate``, or tag attributes are accessed.

    Uses a dynamically-built Pydantic model for validation, derived from the
    same column config that SmartPandas uses.

    Parameters
    ----------
    data : dict
        A single record as a dict (e.g. {"age": 31, "weight": 78.0}).
    config_path : str, optional
        Path to the configuration YAML file.
    config : DataConfig, optional
        DataConfig object to use directly.

    Examples
    --------
    >>> sd = SmartDict({"user_id": "1", "age": 31, ...}, config_path="config.yaml")
    >>> sd.state
    State(name=StateName.RAW, ml_stage=MLStage.INFERENCE)
    >>> sd.validate()
    >>> sd["bmi"] = 24.1  # state updates lazily on next access
    """

    def __init__(
        self,
        data: dict,
        config_path: str | None = None,
        config: DataConfig | None = None,
        state: StateName | str | None = None,
    ):
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided")

        super().__init__(data)
        self.config = config if config is not None else read_config(config_path)
        self.name = self.config.name
        self._pinned_state = StateName(state) if isinstance(state, str) else state
        self._key_hash = hash(frozenset(self.keys()))
        self.state = State.from_data(data=self, config=self.config, state_name=self._pinned_state)
        self.schema = None
        if self.state.name not in [StateName.CORRUPTED, StateName.UNKNOWN]:
            self.schema = build_dict_schema(self.config, self.state)

    def _update_state(self) -> None:
        """Re-infer state and rebuild schema if keys have changed."""
        if self._pinned_state is not None:
            return
        current_hash = hash(frozenset(self.keys()))
        if current_hash != self._key_hash:
            self._key_hash = current_hash
            new_state = State.from_data(data=self, config=self.config)
            if new_state != self.state:
                self.state = new_state
                self.schema = None
                if self.state.name not in [StateName.CORRUPTED, StateName.UNKNOWN]:
                    self.schema = build_dict_schema(self.config, self.state)

    def validate(self, inplace: bool = False) -> dict | None:
        """Validate the dict using the Pydantic schema for the current state.

        Parameters
        ----------
        inplace : bool, default False
            If True, update the SmartDict with coerced values from validation
            and return None. If False, return the validated data as a plain dict.

        Returns
        -------
        dict or None
            The validated data as a plain dict if inplace=False, None otherwise.

        Raises
        ------
        StateError
            If the data is in an invalid state for validation.
        DictValidationError
            If validation fails.
        """
        if self.state.name in [StateName.UNKNOWN, StateName.CORRUPTED]:
            raise StateError(
                f"Cannot validate data in {self.state.name.value} state. "
                "Please check your data and try again."
            )

        try:
            validated = self.schema(**self)
        except ValidationError as e:
            errors = [
                f"{err['loc'][0]}: {err['msg']}" if err.get("loc") else err["msg"]
                for err in e.errors()
            ]
            raise DictValidationError(errors) from e

        coerced = validated.model_dump()
        if inplace:
            dict.update(self, coerced)
            return None

        return coerced

    def __getattribute__(self, name: str) -> Any:
        """Lazily update state when state-sensitive attributes are accessed."""
        if name in _STATE_ATTRS:
            self._update_state()
            if name not in ("state", "validate"):
                # Tag attribute access — return filtered sub-dict
                state = super().__getattribute__("state")
                if name in state.name.incompatibilities or name in state.ml_stage.incompatibilities:
                    return None
                config = super().__getattribute__("config")
                col_names = getattr(config, name)
                return {k: v for k, v in self.items() if k in col_names}
        return super().__getattribute__(name)

    def set_state(self, state: StateName | str | None) -> None:
        """Set or clear the pinned state.

        Parameters
        ----------
        state : StateName, str, or None
            The state to pin to (e.g. "raw", "processed", StateName.RAW).
            Pass None to clear the pin and re-enable automatic inference.
        """
        self._pinned_state = StateName(state) if isinstance(state, str) else state
        self.state = State.from_data(data=self, config=self.config, state_name=self._pinned_state)
        self._key_hash = hash(frozenset(self.keys()))
        self.schema = None
        if self.state.name not in [StateName.CORRUPTED, StateName.UNKNOWN]:
            self.schema = build_dict_schema(self.config, self.state)

    def to_pyspark_schema(self):
        """Return a PySpark StructType schema for the current state.

        Returns
        -------
        pyspark.sql.types.StructType
        """
        return build_pyspark_schema(self.config, self.state)

    def __repr__(self) -> str:
        return f"SmartDict(state={self.state!r}, data={dict(self)})"
