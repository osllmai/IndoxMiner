# import asyncio
# from typing import List, Dict, Any, Union, Tuple, Optional
# from loguru import logger
# import re
# import pandas as pd
# import json

# from .loader import Document
# from .schema import Field, FieldType, OutputFormat
# from .schema import ExtractorSchema
# from .extraction_results import ExtractionResult, ExtractionResults
# from .llms import BaseLLM


# class Extractor:
#     """Data extractor using LLM with validation and concurrent processing.

#     This class handles extraction of structured data from text using a language model,
#     with support for validation, batching, and multiple input formats.

#     Attributes:
#         llm (BaseLLM): Language model provider
#         schema (ExtractorSchema): Extraction schema definition
#         max_concurrent (int): Maximum concurrent extraction operations
#     """

#     def __init__(
#             self,
#             llm: BaseLLM,
#             schema: ExtractorSchema,
#             max_concurrent: int = 3
#     ):
#         self.llm = llm
#         self.schema = schema
#         self.max_concurrent = max_concurrent

#         self.is_async = asyncio.iscoroutinefunction(self.llm.generate)


#     def _validate_field(self, field: Field, value: Any) -> List[str]:
#         """Validate a single field value against its rules.

#         Args:
#             field (Field): Field definition with validation rules
#             value (Any): Value to validate

#         Returns:
#             List[str]: List of validation error messages
#         """
#         errors = []

#         if value is None:
#             if field.required:
#                 errors.append(f"{field.name} is required but missing")
#             return errors

#         if field.rules:
#             rules = field.rules
#             if rules.min_value is not None and value < rules.min_value:
#                 errors.append(f"{field.name} is below minimum value {rules.min_value}")
#             if rules.max_value is not None and value > rules.max_value:
#                 errors.append(f"{field.name} exceeds maximum value {rules.max_value}")
#             if rules.pattern is not None and isinstance(value, str):
#                 if not re.match(rules.pattern, value):
#                     errors.append(f"{field.name} does not match pattern {rules.pattern}")
#             if rules.allowed_values is not None and value not in rules.allowed_values:
#                 errors.append(f"{field.name} contains invalid value")
#             if rules.min_length is not None and len(str(value)) < rules.min_length:
#                 errors.append(f"{field.name} is shorter than minimum length {rules.min_length}")
#             if rules.max_length is not None and len(str(value)) > rules.max_length:
#                 errors.append(f"{field.name} exceeds maximum length {rules.max_length}")

#         return errors
    
#     async def _extract_chunk(self, text: str, chunk_index: int) -> Tuple[int, ExtractionResult]:
#         """Extract data from a single text chunk.
        
#         Args:
#             text (str): Text chunk to process
#             chunk_index (int): Index of the chunk being processed
            
#         Returns:
#             Tuple[int, ExtractionResult]: Chunk index and extraction results
#         """
#         try:

#             prompt = self.schema.to_prompt(text)
            
#             # Handle both sync and async LLMs
#             if self.is_async:
#                 response = await self.llm.generate(prompt)
#             else:
#                 # Run sync generate in thread pool to avoid blocking
#                 loop = asyncio.get_event_loop()
#                 response = await loop.run_in_executor(None, self.llm.generate, prompt)
 
            
#             if self.schema.output_format == OutputFormat.JSON:
#                 def clean_json_response(response_text: str) -> str:
#                     response_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())
#                     lines = []
#                     for line in response_text.split('\n'):
#                         line = re.sub(r'\s*//.*$', '', line.rstrip())
#                         if line:
#                             lines.append(line)
#                     return '\n'.join(lines)

#                 try:
#                     cleaned_response = clean_json_response(response)
#                     logger.debug(f"Cleaned JSON response: {cleaned_response}")

#                     try:
#                         data = json.loads(cleaned_response)
#                     except json.JSONDecodeError as parse_error:
#                         fixed_json = cleaned_response
#                         fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
#                         fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
#                         data = json.loads(fixed_json)

#                     # Handle different data structures
#                     if isinstance(data, list):
#                         # If data is a list, wrap it in an items object
#                         data = {"items": data}
#                     elif isinstance(data, dict):
#                         if not any(isinstance(v, list) for v in data.values()):
#                             # If it's a flat dictionary with no lists, treat as single item
#                             data = {"items": [data]}
#                         else:
#                             # Look for array fields that might contain line items
#                             items = []
#                             common_fields = {}
                            
#                             for key, value in data.items():
#                                 if isinstance(value, list) and all(isinstance(item, dict) for item in value):
#                                     items.extend(value)
#                                 elif not isinstance(value, (list, dict)):
#                                     common_fields[key] = value
                            
#                             if items:
#                                 # Combine common fields with each item
#                                 data = {
#                                     "items": [
#                                         {**common_fields, **item}
#                                         for item in items
#                                     ]
#                                 }
#                             else:
#                                 # No lists found, treat as single item
#                                 data = {"items": [data]}

#                     validation_errors = []
                    
#                     # Validate each item
#                     for i, item in enumerate(data["items"]):
#                         item_errors = []
#                         for field in self.schema.fields:
#                             value = item.get(field.name)
#                             errors = self._validate_field(field, value)
#                             if errors:
#                                 item_errors.extend([f"Item {i + 1}: {error}" for error in errors])
#                         validation_errors.extend(item_errors)

#                     return chunk_index, ExtractionResult(
#                         data=data,
#                         raw_response=response,
#                         validation_errors=validation_errors
#                     )

#                 except json.JSONDecodeError as e:
#                     logger.error(f"JSON parsing error: {e}\nCleaned Response: {cleaned_response}")
#                     return chunk_index, ExtractionResult(
#                         data={},
#                         raw_response=response,
#                         validation_errors=[f"JSON parsing error: {str(e)}"]
#                     )
#             else:
#                 return chunk_index, ExtractionResult(
#                     data={},
#                     raw_response=response,
#                     validation_errors=["Non-JSON formats are returned as raw text"]
#                 )

#         except Exception as e:
#             logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
#             return chunk_index, ExtractionResult(
#                 data={},
#                 raw_response=str(e),
#                 validation_errors=[str(e)]
#             )
        
#     async def extract_single(self, text: str) -> ExtractionResult:
#         """Extract data from a single text.

#         Args:
#             text (str): Text to process

#         Returns:
#             ExtractionResult: Extraction results with validation
#         """
#         _, result = await self._extract_chunk(text, 0)
#         return result
    
    

#     async def extract_multiple(self, documents: List[Document]) -> ExtractionResults:
#         """Extract data from multiple documents concurrently.

#         Args:
#             documents (List[Document]): List of documents to process

#         Returns:
#             ExtractionResults: Combined extraction results with validation
#         """
#         semaphore = asyncio.Semaphore(self.max_concurrent)

#         async def extract_with_semaphore(text: str, index: int) -> Tuple[int, ExtractionResult]:
#             async with semaphore:
#                 return await self._extract_chunk(text, index)

#         tasks = [
#             extract_with_semaphore(doc.page_content, i)
#             for i, doc in enumerate(documents)
#         ]

#         chunk_results = await asyncio.gather(*tasks)
#         chunk_results.sort(key=lambda x: x[0])

#         # Combine results
#         combined_results = ExtractionResults(
#             data=[],
#             raw_responses=[],
#             validation_errors={}
#         )

#         for chunk_index, result in chunk_results:
#             combined_results.data.append(result.data)
#             combined_results.raw_responses.append(result.raw_response)
#             if result.validation_errors:
#                 combined_results.validation_errors[chunk_index] = result.validation_errors

#         return combined_results

#     async def extract(self,
#                       input_data: Union[str, Document, List[Document], Dict[str, List[Document]]]) -> Union[
#         ExtractionResult, ExtractionResults]:
#         """
#         Unified extraction method that handles various input types.

#         Args:
#             input_data: Can be:
#                 - A string (single text)
#                 - A Document object
#                 - A list of Documents
#                 - Output from DocumentProcessor.process() (Dict[str, List[Document]])

#         Returns:
#             ExtractionResult for single inputs or ExtractionResults for multiple documents
#         """
#         if isinstance(input_data, str):
#             return await self.extract_single(input_data)

#         elif isinstance(input_data, Document):
#             return await self.extract_single(input_data.page_content)

#         elif isinstance(input_data, list):
#             return await self.extract_multiple(input_data)

#         elif isinstance(input_data, dict):
#             # Handle DocumentProcessor output
#             all_documents = []
#             for source_documents in input_data.values():
#                 all_documents.extend(source_documents)
#             return await self.extract_multiple(all_documents)

#         else:
#             raise ValueError("Unsupported input type")
#     def extract(self, 
#                 input_data: Union[str, Document, List[Document], Dict[str, List[Document]]]) -> Union[
#         ExtractionResult, ExtractionResults]:
#         """
#         Unified extraction method that works with both sync and async LLMs.
#         """
#         try:
#             loop = asyncio.get_event_loop()
#             if loop.is_running():
#                 # If we're in an async context, return a coroutine
#                 return self._async_extract(input_data)
#             else:
#                 # If we're in a sync context, run the async code
#                 return loop.run_until_complete(self._async_extract(input_data))
#         except RuntimeError:
#             # No event loop exists, create one
#             return asyncio.run(self._async_extract(input_data))
        

#     def to_dataframe(self, result: Union[ExtractionResult, ExtractionResults]) -> Optional[pd.DataFrame]:
#         """Convert extraction result to a pandas DataFrame."""
#         try:
#             if isinstance(result, ExtractionResult):
#                 if 'items' in result.data:
#                     df = pd.DataFrame(result.data['items'])
#                 else:
#                     df = pd.DataFrame([result.data])
#             elif isinstance(result, ExtractionResults):
#                 if any('items' in res for res in result.data):
#                     # Flatten items from all results
#                     items = []
#                     for res in result.data:
#                         if 'items' in res:
#                             items.extend(res['items'])
#                         else:
#                             items.append(res)
#                     df = pd.DataFrame(items)
#                 else:
#                     df = pd.DataFrame(result.data)
#             else:
#                 raise ValueError("Invalid result type")

#             # Clean up the DataFrame
#             df = df.reset_index(drop=True)

#             # Ensure consistent column order based on schema
#             expected_columns = [field.name for field in self.schema.fields]
#             df = df.reindex(columns=expected_columns)

#             # Convert numeric columns to appropriate types
#             numeric_columns = [
#                 field.name for field in self.schema.fields
#                 if field.field_type in [FieldType.FLOAT, FieldType.INTEGER]
#             ]
#             for col in numeric_columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')

#             return df

#         except Exception as e:
#             logger.error(f"Failed to convert to DataFrame: {e}")
#             return None

import asyncio
from typing import List, Dict, Any, Union, Tuple, Optional
from loguru import logger
import re
import pandas as pd
import json

from .loader import Document
from .schema import Field, FieldType, OutputFormat
from .schema import ExtractorSchema
from .extraction_results import ExtractionResult, ExtractionResults
from .llms import BaseLLM


class Extractor:
    """Data extractor using LLM with validation and concurrent processing."""

    def __init__(
            self,
            llm: BaseLLM,
            schema: ExtractorSchema,
            max_concurrent: int = 3
    ):
        self.llm = llm
        self.schema = schema
        self.max_concurrent = max_concurrent
        self.is_async = asyncio.iscoroutinefunction(self.llm.generate)

    def _sync_extract_chunk(self, text: str, chunk_index: int) -> Tuple[int, ExtractionResult]:
        """Synchronous version of extract chunk."""
        try:
            prompt = self.schema.to_prompt(text)
            response = self.llm.generate(prompt)
            return self._process_response(response, chunk_index)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
            return chunk_index, ExtractionResult(
                data={},
                raw_response=str(e),
                validation_errors=[f"Extraction error: {str(e)}"]
            )

    async def _async_extract_chunk(self, text: str, chunk_index: int) -> Tuple[int, ExtractionResult]:
        """Asynchronous version of extract chunk."""
        try:
            prompt = self.schema.to_prompt(text)
            response = await self.llm.generate(prompt)
            return self._process_response(response, chunk_index)
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk_index}: {e}")
            return chunk_index, ExtractionResult(
                data={},
                raw_response=str(e),
                validation_errors=[f"Extraction error: {str(e)}"]
            )
        
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

    def _process_response(self, response: str, chunk_index: int) -> Tuple[int, ExtractionResult]:
        """Process and validate the LLM response."""
        if self.schema.output_format == OutputFormat.JSON:
            try:
                cleaned_response = self._clean_json_response(response)
                logger.debug(f"Cleaned JSON response: {cleaned_response}")

                try:
                    data = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    fixed_json = self._fix_json(cleaned_response)
                    data = json.loads(fixed_json)

                data = self._normalize_json_structure(data)
                validation_errors = self._validate_data(data)

                return chunk_index, ExtractionResult(
                    data=data,
                    raw_response=response,
                    validation_errors=validation_errors
                )

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}\nCleaned Response: {cleaned_response}")
                return chunk_index, ExtractionResult(
                    data={},
                    raw_response=response,
                    validation_errors=[f"JSON parsing error: {str(e)}"]
                )
        else:
            return chunk_index, ExtractionResult(
                data={},
                raw_response=response,
                validation_errors=["Non-JSON formats are returned as raw text"]
            )

    def extract(self, 
                input_data: Union[str, Document, List[Document], Dict[str, List[Document]]]) -> Union[
        ExtractionResult, ExtractionResults]:
        """
        Unified extraction method that handles both sync and async LLMs appropriately.
        """
        if not self.is_async:
            # Synchronous LLM handling
            if isinstance(input_data, str):
                return self._sync_extract_chunk(input_data, 0)[1]
            elif isinstance(input_data, Document):
                return self._sync_extract_chunk(input_data.page_content, 0)[1]
            elif isinstance(input_data, list):
                results = [
                    self._sync_extract_chunk(doc.page_content, i)
                    for i, doc in enumerate(input_data)
                ]
                results.sort(key=lambda x: x[0])
                return ExtractionResults(
                    data=[result.data for _, result in results],
                    raw_responses=[result.raw_response for _, result in results],
                    validation_errors={
                        i: result.validation_errors 
                        for i, (_, result) in enumerate(results) 
                        if result.validation_errors
                    }
                )
            elif isinstance(input_data, dict):
                all_documents = []
                for source_documents in input_data.values():
                    all_documents.extend(source_documents)
                return self.extract(all_documents)
            else:
                raise ValueError("Unsupported input type")
        else:
            # Asynchronous LLM handling
            try:

                try:
                    loop = asyncio.get_running_loop()
                    return self._async_extract(input_data)
                except RuntimeError:
                    # No running event loop, create one
                    return asyncio.run(self._async_extract(input_data))
            except Exception as e:
                logger.error(f"Async extraction failed: {str(e)}")
                raise

    async def _async_extract(self, 
                           input_data: Union[str, Document, List[Document], Dict[str, List[Document]]]) -> Union[
        ExtractionResult, ExtractionResults]:
        """Internal async extraction implementation"""
        if isinstance(input_data, str):
            _, result = await self._async_extract_chunk(input_data, 0)
            return result
        elif isinstance(input_data, Document):
            _, result = await self._async_extract_chunk(input_data.page_content, 0)
            return result
        elif isinstance(input_data, list):
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def extract_with_semaphore(text: str, index: int):
                async with semaphore:
                    return await self._async_extract_chunk(text, index)
                    
            tasks = [
                extract_with_semaphore(doc.page_content, i)
                for i, doc in enumerate(input_data)
            ]
            
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])
            
            return ExtractionResults(
                data=[result.data for _, result in results],
                raw_responses=[result.raw_response for _, result in results],
                validation_errors={
                    i: result.validation_errors 
                    for i, (_, result) in enumerate(results) 
                    if result.validation_errors
                }
            )
        elif isinstance(input_data, dict):
            all_documents = []
            for source_documents in input_data.values():
                all_documents.extend(source_documents)
            return await self._async_extract(all_documents)
        else:
            raise ValueError("Unsupported input type")

    # Helper methods for JSON processing
    def _clean_json_response(self, response_text: str) -> str:
        response_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())
        lines = []
        for line in response_text.split('\n'):
            line = re.sub(r'\s*//.*$', '', line.rstrip())
            if line:
                lines.append(line)
        return '\n'.join(lines)

    def _fix_json(self, json_str: str) -> str:
        fixed_json = json_str
        fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
        fixed_json = re.sub(r'}\s*{', '},{', fixed_json)
        return fixed_json

    def _normalize_json_structure(self, data: Union[Dict, List]) -> Dict:
        if isinstance(data, list):
            data = {"items": data}
        elif isinstance(data, dict):
            if not any(isinstance(v, list) for v in data.values()):
                data = {"items": [data]}
            else:
                items = []
                common_fields = {}
                
                for key, value in data.items():
                    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                        items.extend(value)
                    elif not isinstance(value, (list, dict)):
                        common_fields[key] = value
                
                if items:
                    data = {
                        "items": [
                            {**common_fields, **item}
                            for item in items
                        ]
                    }
                else:
                    data = {"items": [data]}
        return data

    def _validate_data(self, data: Dict) -> List[str]:
        validation_errors = []
        for i, item in enumerate(data["items"]):
            for field in self.schema.fields:
                value = item.get(field.name)
                errors = self._validate_field(field, value)
                if errors:
                    validation_errors.extend([f"Item {i + 1}: {error}" for error in errors])
        return validation_errors
    
    
    def to_dataframe(self, result: Union[ExtractionResult, ExtractionResults]) -> Optional[pd.DataFrame]:
        """Convert extraction result to a pandas DataFrame.
        
        Args:
            result: ExtractionResult or ExtractionResults to convert

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the extracted data, or None if conversion fails
        """
        try:
            if isinstance(result, ExtractionResult):
                if 'items' in result.data:
                    df = pd.DataFrame(result.data['items'])
                else:
                    df = pd.DataFrame([result.data])
            elif isinstance(result, ExtractionResults):
                if any('items' in res for res in result.data):
                    # Flatten items from all results
                    items = []
                    for res in result.data:
                        if 'items' in res:
                            items.extend(res['items'])
                        else:
                            items.append(res)
                    df = pd.DataFrame(items)
                else:
                    df = pd.DataFrame(result.data)
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