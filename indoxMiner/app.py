import asyncio
from typing import List, Dict, Any, Optional, Union, Protocol, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from loguru import logger
import re
from .utils import  Document
from .llms import BaseLLM


class OutputFormat(Enum):
    JSON = "json"
    CSV = "csv"
    TABLE = "table"
    MARKDOWN = "markdown"


class FieldType(Enum):
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
    """Validation rules for fields."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def to_prompt_string(self) -> str:
        """Convert rules to human-readable format."""
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
    """Enhanced field definition with validation."""
    name: str
    description: str
    field_type: FieldType
    required: bool = True
    rules: Optional[ValidationRule] = None
    array_item_type: Optional[FieldType] = None  # For list fields

    def to_prompt_string(self) -> str:
        """Convert field definition to prompt format."""
        type_desc = self.field_type.value
        if self.field_type == FieldType.LIST and self.array_item_type:
            type_desc = f"list of {self.array_item_type.value}s"

        desc = f"{self.name} ({type_desc}{'*' if self.required else ''}): {self.description}"

        if self.rules:
            rules_str = self.rules.to_prompt_string()
            if rules_str:
                desc += f"\n    Validation: {rules_str}"

        return desc


class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers."""

    async def generate(self, prompt: str) -> str:
        ...


@dataclass
class ExtractorSchema:
    """Enhanced schema with better validation and prompting."""
    fields: List[Field]
    output_format: OutputFormat = OutputFormat.JSON
    examples: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None

    def to_prompt(self, text: str) -> str:
        """Generate a detailed extraction prompt."""
        # Build field descriptions
        fields_desc = "\n".join(f"- {field.to_prompt_string()}"
                                for field in self.fields)

        # Build context section
        context_section = f"\nContext:\n{self.context}\n" if self.context else ""

        # Build examples section
        examples_section = ""
        if self.examples:
            examples_json = json.dumps(self.examples, indent=2)
            examples_section = f"\nExamples:\n{examples_json}\n"

        # Build format-specific instructions
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
    """Container for extraction results."""
    data: Dict[str, Any]
    raw_response: str
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


@dataclass
class ExtractionResults:
    """Container for multiple extraction results."""
    combined_data: List[Dict[str, Any]]
    raw_responses: List[str]
    validation_errors: Dict[int, List[str]]

    @property
    def is_valid(self) -> bool:
        return all(not errors for errors in self.validation_errors.values())

    def get_valid_results(self) -> List[Dict[str, Any]]:
        """Return only the valid results."""
        return [data for i, data in enumerate(self.combined_data)
                if not self.validation_errors.get(i, [])]


class Extractor:
    """Enhanced data extractor with automatic chunk handling."""

    def __init__(
            self,
            llm: BaseLLM,
            schema: ExtractorSchema,
            max_concurrent: int = 3  # Limit concurrent extractions
    ):
        self.llm = llm
        self.schema = schema
        self.max_concurrent = max_concurrent

    def _validate_field(self, field: Field, value: Any) -> List[str]:
        """Validate a single field value."""
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
        """Extract data from a single chunk."""
        try:
            prompt = self.schema.to_prompt(text)
            print(prompt)
            response = await self.llm.generate(prompt)
            if self.schema.output_format == OutputFormat.JSON:
                def clean_json_response(response_text: str) -> str:
                    """Clean JSON response by removing comments and fixing common issues."""
                    # Remove markdown code blocks
                    response_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())

                    # Process line by line
                    lines = []
                    for line in response_text.split('\n'):
                        # Remove inline comments
                        line = re.sub(r'\s*//.*$', '', line.rstrip())
                        if line:  # Only add non-empty lines
                            lines.append(line)

                    return '\n'.join(lines)

                try:
                    # Clean and parse the response
                    cleaned_response = clean_json_response(response)
                    logger.debug(f"Cleaned JSON response: {cleaned_response}")

                    try:
                        data = json.loads(cleaned_response)
                    except json.JSONDecodeError as parse_error:
                        # If parsing fails, try to fix common JSON issues
                        fixed_json = cleaned_response
                        # Remove trailing commas before closing brackets
                        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                        # Add missing commas between objects in array
                        fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
                        data = json.loads(fixed_json)

                    # If we got a list, convert it to a dictionary with nested items
                    if isinstance(data, list):
                        data = {
                            "items": data
                        }

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}\nCleaned Response: {cleaned_response}")
                    raise ValueError(f"Could not parse JSON from response: {e}")
            else:
                return chunk_index, ExtractionResult(
                    data={},
                    raw_response=response,
                    validation_errors=["Non-JSON formats are returned as raw text"]
                )

            # Validate all items if it's a list result
            validation_errors = []
            if "items" in data:
                for i, item in enumerate(data["items"]):
                    for field in self.schema.fields:
                        value = item.get(field.name)
                        errors = self._validate_field(field, value)
                        if errors:
                            validation_errors.extend([f"Item {i + 1}: {error}" for error in errors])
            else:
                # Original validation for single object
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
        """Extract from a single piece of text."""
        _, result = await self._extract_chunk(text, 0)
        return result

    async def extract_multiple(self, documents: List[Document]) -> ExtractionResults:
        """
        Extract data from multiple documents/chunks concurrently.

        Args:
            documents: List of Document objects from the DocumentProcessor

        Returns:
            ExtractionResults containing combined data and validation information
        """
        # Create semaphore to limit concurrent extractions
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def extract_with_semaphore(text: str, index: int) -> Tuple[int, ExtractionResult]:
            async with semaphore:
                return await self._extract_chunk(text, index)

        # Process all chunks concurrently with rate limiting
        tasks = [
            extract_with_semaphore(doc.page_content, i)
            for i, doc in enumerate(documents)
        ]

        # Gather results while maintaining order
        chunk_results = await asyncio.gather(*tasks)
        chunk_results.sort(key=lambda x: x[0])  # Sort by chunk index

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
