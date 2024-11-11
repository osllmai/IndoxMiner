from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import re
from datetime import datetime


@dataclass
class AutoDetectedField:
    """Automatically detected field with inferred type and metadata."""

    name: str
    field_type: str
    description: str
    required: bool = False
    rules: Optional[Dict[str, Any]] = None


@dataclass
class AutoExtractionRules:
    """Rules for auto-extraction and validation."""

    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


@dataclass
class AutoSchema:
    """Schema that automatically detects and adapts to document structure."""

    fields: List[AutoDetectedField] = field(default_factory=list)
    detected_structure: Dict[str, Any] = field(default_factory=dict)

    def infer_structure(self, text: str) -> None:
        """Analyze text to infer structure and field types."""
        self.fields = []

        # Detect table-like structures
        if self._looks_like_table(text):
            headers = self._detect_headers(text)
            if headers:
                for header in headers:
                    field_type = self._infer_field_type(header, text)
                    self.fields.append(
                        AutoDetectedField(
                            name=header,
                            field_type=field_type,
                            description=f"Automatically detected {field_type} field from table column",
                            required=True,
                            rules=self._generate_rules(field_type),
                        )
                    )

        # Detect form-like structures
        form_fields = self._detect_form_fields(text)
        for label, sample_value in form_fields.items():
            field_type = self._infer_field_type(label, sample_value)
            self.fields.append(
                AutoDetectedField(
                    name=label,
                    field_type=field_type,
                    description=f"Automatically detected {field_type} field from form",
                    required=True,
                    rules=self._generate_rules(field_type),
                )
            )

    def to_prompt(self, text: str) -> str:
        """Generate extraction prompt based on detected structure."""
        # First analyze the text to detect structure if not already done
        if not self.fields:
            self.infer_structure(text)

        # Build field descriptions
        fields_desc = "\n".join(
            f"- {field.name} ({field.field_type}): {field.description}"
            for field in self.fields
        )

        # Determine if we're dealing with tabular data
        is_tabular = self._looks_like_table(text)
        table_instruction = (
            """
- Extract data in a tabular format
- Preserve column headers and row relationships
- Return as an array of objects"""
            if is_tabular
            else ""
        )

        return f"""Task: Extract structured information from the given text using automatic field detection.

Detected Fields:
{fields_desc}

Extraction Requirements:
1. Extract all detected fields maintaining their original names
2. Use appropriate data types for each field:
   - Dates in ISO format (YYYY-MM-DD)
   - Numbers as numeric values (not strings)
   - Boolean values as true/false
   - Lists as arrays
   - Nested data as objects
3. Preserve any detected relationships between fields
4. Return data in JSON format{table_instruction}

Text to analyze:
{text}"""

    def _looks_like_table(self, text: str) -> bool:
        """Detect if text contains table-like structure."""
        lines = text.split("\n")
        if len(lines) < 2:
            return False

        # Check for common table indicators
        has_delimiter_row = any(
            line.count("|") > 1 or line.count("\t") > 1 for line in lines
        )
        has_consistent_spacing = self._check_consistent_spacing(lines)
        has_header_indicators = any(
            line.count("-") > 3 or line.count("=") > 3 for line in lines
        )

        return has_delimiter_row or has_consistent_spacing or has_header_indicators

    def _check_consistent_spacing(self, lines: List[str]) -> bool:
        """Check if lines have consistent spacing pattern."""
        if len(lines) < 2:
            return False

        # Get positions of whitespace chunks
        space_positions = []
        for line in lines[:3]:  # Check first few lines
            positions = [i for i, char in enumerate(line) if char.isspace()]
            if positions:
                space_positions.append(positions)

        # Check if space positions are consistent
        if len(space_positions) > 1:
            return any(
                abs(len(pos1) - len(pos2)) <= 1
                for pos1, pos2 in zip(space_positions[:-1], space_positions[1:])
            )

        return False

    def _detect_headers(self, text: str) -> List[str]:
        """Detect column headers from table-like text."""
        lines = text.split("\n")
        potential_headers = []

        for i, line in enumerate(lines[:3]):  # Check first few lines
            # Split by common delimiters
            cells = re.split(r"\s{2,}|\t|\|", line.strip())
            cells = [cell.strip() for cell in cells if cell.strip()]

            # Header characteristics
            looks_like_header = all(
                word[0].isupper() for word in cells if word
            ) and not any(cell.replace(".", "").isdigit() for cell in cells)

            if looks_like_header:
                potential_headers = cells
                break

        return potential_headers

    def _detect_form_fields(self, text: str) -> Dict[str, str]:
        """Detect form-like field labels and sample values."""
        fields = {}
        lines = text.split("\n")

        for line in lines:
            # Look for label-value patterns
            matches = re.finditer(r"([A-Za-z][A-Za-z\s]+)[\s:]+([^:]+)(?=\s*|$)", line)
            for match in matches:
                label = match.group(1).strip()
                value = match.group(2).strip()
                if label and value:
                    fields[label] = value

        return fields

    def _infer_field_type(self, label: str, sample: str) -> str:
        """Infer field type from label and sample value."""
        # Check for date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}/\d{2}/\d{4}",
            r"\d{2}\.\d{2}\.\d{4}",
        ]
        if any(re.search(pattern, str(sample)) for pattern in date_patterns):
            return "date"

        # Check for numeric patterns
        if isinstance(sample, (int, float)) or (
            isinstance(sample, str)
            and re.match(r"^-?\d*\.?\d+$", sample.replace(",", ""))
        ):
            return "number"

        # Check for boolean indicators
        bool_values = {"true", "false", "yes", "no", "y", "n"}
        if str(sample).lower() in bool_values:
            return "boolean"

        # Check for list indicators
        if isinstance(sample, list) or (
            isinstance(sample, str) and ("," in sample or ";" in sample)
        ):
            return "list"

        # Default to string
        return "string"

    def _generate_rules(self, field_type: str) -> AutoExtractionRules:
        """Generate appropriate validation rules based on field type."""
        rules = AutoExtractionRules()

        if field_type == "string":
            rules.min_length = 1
            rules.max_length = 1000
        elif field_type == "number":
            rules.min_value = float("-inf")
            rules.max_value = float("inf")
        elif field_type == "date":
            rules.pattern = r"^\d{4}-\d{2}-\d{2}$"
        elif field_type == "boolean":
            rules.allowed_values = [True, False]

        return rules

    def validate_extraction(self, data: Dict[str, Any]) -> List[str]:
        """Validate extracted data against inferred rules."""
        errors = []

        for field in self.fields:
            value = data.get(field.name)
            if field.required and value is None:
                errors.append(f"{field.name} is required but missing")
                continue

            if value is not None and field.rules:
                rules = field.rules
                if rules.min_length and len(str(value)) < rules.min_length:
                    errors.append(
                        f"{field.name} is shorter than minimum length {rules.min_length}"
                    )
                if rules.max_length and len(str(value)) > rules.max_length:
                    errors.append(
                        f"{field.name} exceeds maximum length {rules.max_length}"
                    )
                if rules.pattern and not re.match(rules.pattern, str(value)):
                    errors.append(f"{field.name} does not match expected pattern")
                if rules.min_value and value < rules.min_value:
                    errors.append(
                        f"{field.name} is below minimum value {rules.min_value}"
                    )
                if rules.max_value and value > rules.max_value:
                    errors.append(
                        f"{field.name} exceeds maximum value {rules.max_value}"
                    )
                if rules.allowed_values and value not in rules.allowed_values:
                    errors.append(f"{field.name} contains invalid value")

        return errors
