from .app import (
    Extractor,
    ExtractorSchema,
    ExtractionResult,
    Field,
    ValidationRule,
    OutputFormat,
    FieldType
)

from .loader import (
    DocumentProcessor,
    ProcessingConfig,
    DocumentType,
    Document
)

from .llms import (
    OpenAi,
    Anthropic,
    # Ollama,
)
__all__ = [
    # Extractor related
    "Extractor",
    "ExtractorSchema",
    "ExtractionResult",
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
    # "Ollama",
]

# Package metadata
__version__ = "0.0.0"
__author__ = "IndoxMiner Team"
__description__ = "A comprehensive document extraction and processing library"
