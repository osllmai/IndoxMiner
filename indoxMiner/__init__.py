# __init__.py

from .extractor import Extractor
from .extractor_schema import ExtractorSchema, Schema
from .extraction_results import ExtractionResult, ExtractionResults
from .schema import Field, ValidationRule, OutputFormat, FieldType

from .loader import DocumentProcessor, ProcessingConfig, DocumentType, Document

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
    "DocumentType",
    "Document",

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
