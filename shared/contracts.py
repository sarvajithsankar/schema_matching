"""
Shared contracts and validation for column profiles.
Defines the ColumnProfile dataclass and validation against column_profile_schema.json.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

SCHEMA_PATH = Path(__file__).resolve().parent / "column_profile_schema.json"


@dataclass
class ColumnProfile:
    """
    Standard column profile contract shared across name and value pipelines.

    Attributes
    ----------
    column_id : str
        Globally unique column identifier (e.g., 'DB1.Users.id').
    schema_name : str
        Source schema or table name.
    normalized_tokens : List[str]
        Tokens derived from column name normalization and abbreviation expansion.
    value_profile : Dict[str, Any]
        Statistical and pattern summary produced by value profiler.
    sampled_values : List[Any]
        Sample of non-null values.
    name_embedding : Optional[List[float]]
        Optional dense embedding vector for semantic name similarity.
    raw_name : Optional[str]
        Original raw column name.
    data_type : Optional[str]
        Inferred data type.
    """

    column_id: str
    schema_name: str
    normalized_tokens: List[str]
    value_profile: Dict[str, Any]
    sampled_values: List[Any] = field(default_factory=list)
    name_embedding: Optional[List[float]] = None
    raw_name: Optional[str] = None
    data_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ColumnProfile instance to dictionary."""
        d = asdict(self)
        # Ensure sample_values alias is provided for backwards compatibility
        d["sample_values"] = list(self.sampled_values)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnProfile":
        """Instantiate ColumnProfile from a dictionary."""
        col_id = data.get("column_id", "")
        schema_name = data.get("schema_name")
        if not schema_name and "." in col_id:
            schema_name = col_id.split(".")[0]
        elif not schema_name:
            schema_name = "default"

        sampled = (
            data.get("sampled_values")
            if "sampled_values" in data
            else data.get("sample_values", data.get("value_sample", []))
        )
        if sampled is None:
            sampled = []

        tokens = data.get("normalized_tokens") or []
        vp = data.get("value_profile") or {}
        embedding = data.get("name_embedding")

        return cls(
            column_id=col_id,
            schema_name=schema_name,
            normalized_tokens=list(tokens),
            value_profile=dict(vp),
            sampled_values=list(sampled),
            name_embedding=list(embedding) if embedding is not None else None,
            raw_name=data.get("raw_name"),
            data_type=data.get("data_type") or vp.get("dtype"),
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style key access."""
        if hasattr(self, key):
            return getattr(self, key)
        if key == "sample_values" or key == "value_sample":
            return self.sampled_values
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dict-style get access."""
        try:
            return self[key]
        except KeyError:
            return default


def load_schema() -> Dict[str, Any]:
    """Load the JSON schema definition for ColumnProfile."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_instance_pure_python(instance: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate a dictionary instance against the schema using pure Python."""
    if not isinstance(instance, dict):
        raise TypeError(f"ColumnProfile instance must be a dict, got {type(instance).__name__}")

    required_fields = schema.get("required", [])
    for req in required_fields:
        if req not in instance:
            # Check for allowed aliases
            if req == "schema_name" and "." in instance.get("column_id", ""):
                continue
            if req == "sampled_values" and ("sample_values" in instance or "value_sample" in instance):
                continue
            raise ValueError(f"Missing required field: '{req}'")

    properties = schema.get("properties", {})
    if "column_id" in instance and not isinstance(instance["column_id"], str):
        raise TypeError("Field 'column_id' must be a string")

    schema_name = instance.get("schema_name")
    if schema_name is not None and not isinstance(schema_name, str):
        raise TypeError("Field 'schema_name' must be a string")

    tokens = instance.get("normalized_tokens")
    if tokens is not None and not isinstance(tokens, list):
        raise TypeError("Field 'normalized_tokens' must be a list of strings")

    sampled = instance.get("sampled_values", instance.get("sample_values", instance.get("value_sample")))
    if sampled is not None and not isinstance(sampled, list):
        raise TypeError("Field 'sampled_values' must be a list")

    vp = instance.get("value_profile")
    if vp is not None and not isinstance(vp, dict):
        raise TypeError("Field 'value_profile' must be a dictionary")

    embedding = instance.get("name_embedding")
    if embedding is not None and not isinstance(embedding, list):
        raise TypeError("Field 'name_embedding' must be a list or None")

    return True


def validate_column_profile(profile: Union[Dict[str, Any], ColumnProfile]) -> bool:
    """
    Validate a column profile dictionary or ColumnProfile object against the JSON schema.
    Uses jsonschema if installed; otherwise falls back to a pure-Python validator.
    """
    schema = load_schema()
    instance = profile.to_dict() if isinstance(profile, ColumnProfile) else dict(profile)

    try:
        from jsonschema import validate as json_validate
        json_validate(instance=instance, schema=schema)
        return True
    except ImportError:
        return _validate_instance_pure_python(instance, schema)


if __name__ == "__main__":
    test_profile = ColumnProfile(
        column_id="test.customer.customer_id",
        schema_name="customer",
        raw_name="cust_id",
        normalized_tokens=["customer", "identifier"],
        data_type="string",
        sampled_values=["C001", "C002"],
        name_embedding=[0.1, 0.2, 0.3],
        value_profile={"dtype": "string"},
    )

    validate_column_profile(test_profile)
    print("[OK] Column profile is valid")