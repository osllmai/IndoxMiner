import asyncio
from typing import List, Dict, Any, Union, Tuple
from loguru import logger
import re
import pandas as pd
from .schema import Field, FieldType
from .extractor_schema import ExtractorSchema
from .extraction_results import ExtractionResult, ExtractionResults
from .utils import Document
from .llms import BaseLLM
import json

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