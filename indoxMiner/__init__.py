# __init__.py

from .extractor import Extractor
from .schema import ExtractorSchema, Schema, OutputFormat
from .extraction_results import ExtractionResult, ExtractionResults
from .fields import Field, ValidationRule, FieldType

from .loader import DocumentProcessor, ProcessingConfig

from .llms import OpenAi, Anthropic, IndoxApi, AsyncIndoxApi, AsyncOpenAi

__all__ = [
    # Extractor and schema related
    "Extractor",
    "ExtractorSchema",
    "Schema",  # For accessing predefined schemas like Passport, Invoice, etc.
    "ExtractionResult",
    "ExtractionResults",
    "Field",
    "ValidationRule",
    "OutputFormat",
    "FieldType",

    # Document processing related
    "DocumentProcessor",
    "ProcessingConfig",

    # llms
    "OpenAi",
    "Anthropic",
    "IndoxApi",
    "AsyncIndoxApi",
    "AsyncOpenAi"
]

# Package metadata
__version__ = "0.0.0"
__author__ = "IndoxMiner Team"
__description__ = "A comprehensive document extraction and processing library"
