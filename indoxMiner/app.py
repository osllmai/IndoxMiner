import asyncio
from typing import List, Dict, Any, Optional, Union, Protocol, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from loguru import logger
import re
from .utils import Document
from .llms import BaseLLM


class OutputFormat(Enum):
    """Supported output formats for data extraction.

    Defines the available formats for structuring extracted data:
    - JSON: JavaScript Object Notation format
    - CSV: Comma-Separated Values format
    - TABLE: Formatted table structure
    - MARKDOWN: Markdown text format
    """
    JSON = "json"
    CSV = "csv"
    TABLE = "table"
    MARKDOWN = "markdown"


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


@dataclass
class ExtractorSchema:
    """Schema definition for data extraction with validation and formatting.

    Attributes:
        fields (List[Field]): List of fields to extract
        output_format (OutputFormat): Desired output format
        examples (List[Dict[str, Any]], optional): Example extractions
        context (str, optional): Additional context for extraction
    """
    fields: List[Field]
    output_format: OutputFormat = OutputFormat.JSON
    examples: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None

    def to_prompt(self, text: str) -> str:
        """Generate extraction prompt based on schema.

        Args:
            text (str): Source text for extraction

        Returns:
            str: Formatted prompt for LLM extraction
        """
        fields_desc = "\n".join(f"- {field.to_prompt_string()}"
                                for field in self.fields)

        context_section = f"\nContext:\n{self.context}\n" if self.context else ""

        examples_section = ""
        if self.examples:
            examples_json = json.dumps(self.examples, indent=2)
            examples_section = f"\nExamples:\n{examples_json}\n"

        format_instructions = {
            OutputFormat.JSON: "Format as a JSON object. Use null for missing values.",
            OutputFormat.CSV: "Format as CSV. Use empty string for missing values.",
            OutputFormat.TABLE: "Format as a markdown table. Use empty cells for missing values.",
            OutputFormat.MARKDOWN: "Format as markdown. Use appropriate markdown syntax."
        }
        return f"""Task: Extract structured information from the given text according to the following schema.

        Fields to extract:
        {fields_desc}{context_section}{examples_section}

        Output Requirements:
        1. Extract ONLY the specified fields
        2. Follow the exact field names provided
        3. Use {self.output_format.value} format
        4. {format_instructions[self.output_format]}
        5. If a required field cannot be found, use null/empty values
        6. Validate all values against provided rules
        7. For dates, use ISO format (YYYY-MM-DD)
        8. For lists, provide values in a consistent format
        9. CRITICAL: Return ONLY the {self.output_format.value} output - no explanations, comments, or additional text before or after
        10. CRITICAL: Do not include explanation of what was extracted
        11. CRITICAL: Do not include ```{self.output_format.value} tags or backticks

        Text to analyze:
        {text}

        Return the pure {self.output_format.value} output now:"""


@dataclass
class ExtractionResult:
    """Container for single extraction result with validation.

    Attributes:
        data (Dict[str, Any]): Extracted data
        raw_response (str): Original LLM response
        validation_errors (List[str]): List of validation errors
    """
    data: Dict[str, Any]
    raw_response: str
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if extraction passed validation.

        Returns:
            bool: True if no validation errors, False otherwise
        """
        return len(self.validation_errors) == 0


@dataclass
class ExtractionResults:
    """Container for multiple extraction results with validation.

    Attributes:
        combined_data (List[Dict[str, Any]]): List of extracted data
        raw_responses (List[str]): Original LLM responses
        validation_errors (Dict[int, List[str]]): Validation errors by index
    """
    combined_data: List[Dict[str, Any]]
    raw_responses: List[str]
    validation_errors: Dict[int, List[str]]

    @property
    def is_valid(self) -> bool:
        """Check if all extractions passed validation.

        Returns:
            bool: True if no validation errors across all results
        """
        return all(not errors for errors in self.validation_errors.values())

    def get_valid_results(self) -> List[Dict[str, Any]]:
        """Get list of results that passed validation.

        Returns:
            List[Dict[str, Any]]: Valid extraction results
        """
        return [data for i, data in enumerate(self.combined_data)
                if not self.validation_errors.get(i, [])]


class Extractor:
    """Data extractor using LLM with validation and concurrent processing.

    This class handles extraction of structured data from text using a language model,
    with support for validation, batching, and multiple input formats.

    Attributes:
        llm (BaseLLM): Language model provider
        schema (ExtractorSchema): Extraction schema definition
        max_concurrent (int): Maximum concurrent extraction operations
    """

    def __init__(
            self,
            llm: BaseLLM,
            schema: ExtractorSchema,
            max_concurrent: int = 3
    ):
        self.llm = llm
        self.schema = schema
        self.max_concurrent = max_concurrent

    def _validate_field(self, field: Field, value: Any) -> List[str]:
        """Validate a single field value against its rules.

        Args:
            field (Field): Field definition with validation rules
            value (Any): Value to validate

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        if value is None:
            if field.required:
                errors.append(f"{field.name} is required but missing")
            return errors

        if field.rules:
            rules = field.rules
            if rules.min_value is not None and value < rules.min_value:
                errors.append(f"{field.name} is below minimum value {rules.min_value}")
            if rules.max_value is not None and value > rules.max_value:
                errors.append(f"{field.name} exceeds maximum value {rules.max_value}")
            if rules.pattern is not None and isinstance(value, str):
                if not re.match(rules.pattern, value):
                    errors.append(f"{field.name} does not match pattern {rules.pattern}")
            if rules.allowed_values is not None and value not in rules.allowed_values:
                errors.append(f"{field.name} contains invalid value")
            if rules.min_length is not None and len(str(value)) < rules.min_length:
                errors.append(f"{field.name} is shorter than minimum length {rules.min_length}")
            if rules.max_length is not None and len(str(value)) > rules.max_length:
                errors.append(f"{field.name} exceeds maximum length {rules.max_length}")

        return errors

    async def _extract_chunk(self, text: str, chunk_index: int) -> Tuple[int, ExtractionResult]:
        """Extract data from a single text chunk.

        Args:
            text (str): Text chunk to process
            chunk_index (int): Index of the chunk

        Returns:
            Tuple[int, ExtractionResult]: Chunk index and extraction results
        """
        try:
            prompt = self.schema.to_prompt(text)
            response = await self.llm.generate(prompt)
            if self.schema.output_format == OutputFormat.JSON:
                def clean_json_response(response_text: str) -> str:
                    response_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())
                    lines = []
                    for line in response_text.split('\n'):
                        line = re.sub(r'\s*//.*$', '', line.rstrip())
                        if line:
                            lines.append(line)
                    return '\n'.join(lines)

                try:
                    cleaned_response = clean_json_response(response)
                    logger.debug(f"Cleaned JSON response: {cleaned_response}")

                    try:
                        data = json.loads(cleaned_response)
                    except json.JSONDecodeError as parse_error:
                        fixed_json = cleaned_response
                        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                        fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
                        data = json.loads(fixed_json)

                    if isinstance(data, list):
                        data = {"items": data}

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}\nCleaned Response: {cleaned_response}")
                    raise ValueError(f"Could not parse JSON from response: {e}")
            else:
                return chunk_index, ExtractionResult(
                    data={},
                    raw_response=response,
                    validation_errors=["Non-JSON formats are returned as raw text"]
                )

            validation_errors = []
            if "items" in data:
                for i, item in enumerate(data["items"]):
                    for field in self.schema.fields:
                        value = item.get(field.name)
                        errors = self._validate_field(field, value)
                        if errors:
                            validation_errors.extend([f"Item {i + 1}: {error}" for error in errors])
            else:
                for field in self.schema.fields:
                    value = data.get(field.name)
                    errors = self._validate_field(field, value)
                    validation_errors.extend(errors)

            return chunk_index, ExtractionResult(
                data=data,
                raw_response=response,
                validation_errors=validation_errors
            )

        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
            return chunk_index, ExtractionResult(
                data={},
                raw_response=str(e),
                validation_errors=[str(e)]
            )

    async def extract_single(self, text: str) -> ExtractionResult:
        """Extract data from a single text.

        Args:
            text (str): Text to process

        Returns:
            ExtractionResult: Extraction results with validation
        """
        _, result = await self._extract_chunk(text, 0)
        return result

    async def extract_multiple(self, documents: List[Document]) -> ExtractionResults:
        """Extract data from multiple documents concurrently.

        Args:
            documents (List[Document]): List of documents to process

        Returns:
            ExtractionResults: Combined extraction results with validation
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_with_semaphore(text: str, index: int) -> Tuple[int, ExtractionResult]:
            async with semaphore:
                return await self._extract_chunk(text, index)

        tasks = [
            extract_with_semaphore(doc.page_content, i)
            for i, doc in enumerate(documents)
        ]

        chunk_results = await asyncio.gather(*tasks)
        chunk_results.sort(key=lambda x: x[0])

        # Combine results
        combined_results = ExtractionResults(
            combined_data=[],
            raw_responses=[],
            validation_errors={}
        )

        for chunk_index, result in chunk_results:
            combined_results.combined_data.append(result.data)
            combined_results.raw_responses.append(result.raw_response)
            if result.validation_errors:
                combined_results.validation_errors[chunk_index] = result.validation_errors

        return combined_results

    async def extract(self,
                      input_data: Union[str, Document, List[Document], Dict[str, List[Document]]]) -> Union[
        ExtractionResult, ExtractionResults]:
        """
        Unified extraction method that handles various input types.

        Args:
            input_data: Can be:
                - A string (single text)
                - A Document object
                - A list of Documents
                - Output from DocumentProcessor.process() (Dict[str, List[Document]])

        Returns:
            ExtractionResult for single inputs or ExtractionResults for multiple documents
        """
        if isinstance(input_data, str):
            return await self.extract_single(input_data)

        elif isinstance(input_data, Document):
            return await self.extract_single(input_data.page_content)

        elif isinstance(input_data, list):
            return await self.extract_multiple(input_data)

        elif isinstance(input_data, dict):
            # Handle DocumentProcessor output
            all_documents = []
            for source_documents in input_data.values():
                all_documents.extend(source_documents)
            return await self.extract_multiple(all_documents)

        else:
            raise ValueError("Unsupported input type")

    def to_dataframe(self, result: Union[ExtractionResult, ExtractionResults]) -> Optional[pd.DataFrame]:
        """Convert extraction result to a pandas DataFrame."""
        try:
            if isinstance(result, ExtractionResult):
                if 'items' in result.data:
                    df = pd.DataFrame(result.data['items'])
                else:
                    df = pd.DataFrame([result.data])
            elif isinstance(result, ExtractionResults):
                if any('items' in res for res in result.combined_data):
                    # Flatten items from all results
                    items = []
                    for res in result.combined_data:
                        if 'items' in res:
                            items.extend(res['items'])
                        else:
                            items.append(res)
                    df = pd.DataFrame(items)
                else:
                    df = pd.DataFrame(result.combined_data)
            else:
                raise ValueError("Invalid result type")

            # Clean up the DataFrame
            df = df.reset_index(drop=True)

            # Ensure consistent column order based on schema
            expected_columns = [field.name for field in self.schema.fields]
            df = df.reindex(columns=expected_columns)

            # Convert numeric columns to appropriate types
            numeric_columns = [
                field.name for field in self.schema.fields
                if field.field_type in [FieldType.FLOAT, FieldType.INTEGER]
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Failed to convert to DataFrame: {e}")
            return None
