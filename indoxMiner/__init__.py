from .data_extraction.extractor import Extractor
from .data_extraction.schema import ExtractorSchema, Schema
from .data_extraction.extraction_results import ExtractionResult, ExtractionResults
from .data_extraction.fields import Field, ValidationRule, FieldType
from .data_extraction.loader import DocumentProcessor, ProcessingConfig

from .data_extraction.llms import (
    OpenAi,
    Anthropic,
    NerdTokenApi,
    AsyncNerdTokenApi,
    AsyncOpenAi,
    Ollama,
)

# Importing IndoxObjectDetector class for object detection
from .object_detection.app import IndoxObjectDetector

__all__ = [
    # Extractor and schema related
    "Extractor",
    "ExtractorSchema",
    "Schema",  # For accessing predefined schemas like Passport, Invoice, etc.
    "ExtractionResult",
    "ExtractionResults",
    "Field",
    "ValidationRule",
    "FieldType",
    # Document processing related
    "DocumentProcessor",
    "ProcessingConfig",
    # llms
    "OpenAi",
    "Anthropic",
    "NerdTokenApi",
    "AsyncNerdTokenApi",
    "AsyncOpenAi",
    "Ollama",
    # Indox Object Detection
    "IndoxObjectDetector",
]

# Package metadata
__version__ = "0.0.9"
__author__ = "IndoxMiner Team"
__description__ = "A comprehensive document extraction and processing library"
