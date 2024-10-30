# IndoxMiner

IndoxMiner is a powerful Python library for extracting structured information from unstructured text and documents using Large Language Models (LLM). It provides a flexible schema-based approach to define and validate the information you want to extract.

## Features

- 🔍 Extract structured data from text and PDFs
- 📄 Support for multiple document formats
- 🔧 Customizable extraction schemas
- ✅ Built-in validation rules
- 📊 Easy conversion to pandas DataFrames
- 🤖 Integration with OpenAI models
- 🎯 Type-safe field definitions
- 🔄 Async support for better performance

## Installation

```bash
pip install indoxminer
```

## Quick Start

### Basic Text Extraction

```python
from indoxminer import (
    ExtractorSchema,
    Field,
    FieldType,
    ValidationRule,
    OutputFormat,
    Extractor,
    OpenAi
)

# Initialize OpenAI
llm_extractor = OpenAi(api_key="your-api-key", model="gpt-4-mini")

# Define extraction schema
schema = ExtractorSchema(
    fields=[
        Field(
            name="product_name",
            description="Product name",
            field_type=FieldType.STRING,
            rules=ValidationRule(min_length=2)
        ),
        Field(
            name="price",
            description="Price in USD",
            field_type=FieldType.FLOAT,
            rules=ValidationRule(min_value=0)
        )
    ]
)

# Create extractor
extractor = Extractor(llm=llm_extractor, schema=schema)

# Extract information
text = """
MacBook Pro 16-inch with M2 chip
Price: $2,399.99
In stock: Yes
"""
result = await extractor.extract(text)

# Convert to DataFrame
df = extractor.to_dataframe(result)
```

### PDF Document Processing

```python
from indoxminer import DocumentProcessor, ProcessingConfig

# Initialize document processor
processor = DocumentProcessor(["invoice.pdf"])

# Process documents with configuration
documents = processor.process(
    config=ProcessingConfig(
        hi_res_pdf=True
    )
)

# Define complex schema
schema = ExtractorSchema(
    fields=[
        Field(
            name="bill_to",
            description="Bill To",
            field_type=FieldType.STRING,
            rules=ValidationRule(min_length=2)
        ),
        Field(
            name="date",
            description="date",
            field_type=FieldType.DATE,
        ),
        Field(
            name="amount",
            description="price in usd",
            field_type=FieldType.FLOAT,
        ),
    ],
    output_format=OutputFormat.JSON
)

# Extract information
results = await extractor.extract(documents)

# Handle results and validation
valid_data = results.get_valid_results()

if not results.is_valid:
    for chunk_idx, errors in results.validation_errors.items():
        print(f"Chunk {chunk_idx} has errors: {errors}")
```

## Core Components

### ExtractorSchema

Defines the structure of data to be extracted:
- `fields`: List of Field objects defining what to extract
- `output_format`: Desired output format (JSON, etc.)

### Field

Defines individual data fields to extract:
- `name`: Field identifier
- `description`: Field description for the LLM
- `field_type`: Data type (STRING, FLOAT, INTEGER, DATE)
- `rules`: Validation rules for the field

### ValidationRule

Sets constraints for extracted data:
- `min_length`: Minimum string length
- `max_length`: Maximum string length
- `min_value`: Minimum numeric value
- `max_value`: Maximum numeric value
- `pattern`: Regex pattern for validation

### DocumentProcessor

Handles document processing:
- Supports multiple file formats
- Configurable processing options
- High-resolution PDF support

## Configuration

### ProcessingConfig

```python
config = ProcessingConfig(
    hi_res_pdf=True,  # Enable high-resolution PDF processing
    # Add other configuration options as needed
)
```

## Error Handling

The library provides comprehensive validation and error handling:
- Validation errors are collected per chunk
- Easy access to valid and invalid results
- Detailed error messages for debugging

## Output Formats

Supported output formats:
- JSON
- Pandas DataFrame
- Raw dictionary

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Support

For issues and feature requests, please use the GitHub issue tracker.