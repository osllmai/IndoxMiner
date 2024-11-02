from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any


@dataclass
class ValidationRule:
    """Validation rules for ensuring data quality in extracted fields.

    Attributes:
        min_value (float, optional): Minimum allowed numeric value
        max_value (float, optional): Maximum allowed numeric value
        pattern (str, optional): Regex pattern for string validation
        allowed_values (List[Any], optional): List of valid values
        min_length (int, optional): Minimum length for string fields
        max_length (int, optional): Maximum length for string fields
    """
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def to_prompt_string(self) -> str:
        """Convert validation rules to a human-readable format.

        Returns:
            str: Semicolon-separated string of active validation rules
        """
        rules = []
        if self.min_value is not None:
            rules.append(f"minimum value: {self.min_value}")
        if self.max_value is not None:
            rules.append(f"maximum value: {self.max_value}")
        if self.pattern is not None:
            rules.append(f"must match pattern: {self.pattern}")
        if self.allowed_values is not None:
            rules.append(f"must be one of: {', '.join(map(str, self.allowed_values))}")
        if self.min_length is not None:
            rules.append(f"minimum length: {self.min_length}")
        if self.max_length is not None:
            rules.append(f"maximum length: {self.max_length}")
        return "; ".join(rules)


class FieldType(Enum):
    """Data types supported for field extraction.

    Defines the possible data types that can be extracted:
    - Standard types: STRING, INTEGER, FLOAT, BOOLEAN, DATE
    - Complex types: LIST
    - Specialized types: EMAIL, PHONE, URL
    """
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    LIST = "list"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"


@dataclass
class Field:
    """Field definition for data extraction with validation rules.

    Attributes:
        name (str): Field identifier
        description (str): Human-readable field description
        field_type (FieldType): Data type of the field
        required (bool): Whether the field must have a value
        rules (ValidationRule, optional): Validation rules for the field
        array_item_type (FieldType, optional): For LIST fields, the type of items
    """
    name: str
    description: str
    field_type: FieldType
    required: bool = True
    rules: Optional[ValidationRule] = None
    array_item_type: Optional[FieldType] = None

    def to_prompt_string(self) -> str:
        """Convert field definition to prompt format.

        Returns:
            str: Human-readable field description including type and rules
        """
        type_desc = self.field_type.value
        if self.field_type == FieldType.LIST and self.array_item_type:
            type_desc = f"list of {self.array_item_type.value}s"

        desc = f"{self.name} ({type_desc}{'*' if self.required else ''}): {self.description}"

        if self.rules:
            rules_str = self.rules.to_prompt_string()
            if rules_str:
                desc += f"\n    Validation: {rules_str}"

        return desc
