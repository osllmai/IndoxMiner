# IndoxMiner: Extracting Structured Data from Images

IndoxMiner provides a powerful and flexible way to extract structured data from unstructured text within images. Using OCR (Optical Character Recognition) to convert image content to text, and LLMs (Large Language Models) to interpret and extract specific fields, IndoxMiner simplifies data extraction from images like invoices, receipts, ID cards, and more.

## Key Features

- 📸 **Image to Structured Data**: Extract information from images and convert it into structured formats.
- 🔠 **OCR Integration**: Supports PaddleOCR for text extraction from images.
- 🔍 **Custom Extraction Schemas**: Define and validate the data fields you want to extract.
- ✅ **Built-in Validation Rules**: Ensures data accuracy with customizable validation options.
- 📊 **Easy Conversion to DataFrames**: Seamlessly convert results to pandas DataFrames.
- 🤖 **LLM Integration**: Use OpenAI, IndoxApi, and other LLMs for advanced text interpretation.
- 🔄 **Async Support**: Process multiple images concurrently for optimized performance.

## Installation

To set up IndoxMiner, clone the repository and install dependencies:

```bash
pip install indoxminer
```
You will also need an OCR library, such as PaddleOCR, to handle the image-to-text conversion.

## Quick Start

### Step 1: Set up the OCR Processor and LLM

```python
from indoxminer import OCRProcessor, Extractor, Schema, AsyncIndoxApi

# Initialize OCR processor (e.g., PaddleOCR)
ocr_processor = OCRProcessor(model="paddle")

# Initialize IndoxMiner API with AsyncIndoxApi
llm_extractor = AsyncIndoxApi(api_key="your-api-key")
```

### Step 2: Define Image Paths and Schema

Define the images to process and select a predefined schema or create a custom one.

```python
# List of images to process
image_paths = ["path/to/invoice1.png", "path/to/invoice2.png"]

# Choose a predefined schema or define your own
schema = Schema.Invoice  # Or Schema.Receipt, Schema.Passport, etc.
```

### Step 3: Process Images and Extract Data

Process each image with OCR to extract the text, then use IndoxMiner to extract structured data according to the schema.

```python
from indoxminer.utils import Document

# Convert images to text using OCR
documents = []
for path in image_paths:
    text = ocr_processor.extract_text(path)
    documents.append(Document(page_content=text, metadata={"filename": path}))

# Create an extractor with the LLM and schema
extractor = Extractor(llm=llm_extractor, schema=schema)

# Run the extraction
extraction_results = await extractor.extract_multiple(documents)

```

### Step 4: Handle Results and Convert to DataFrame

Process each image with OCR to extract the text, then use IndoxMiner to extract structured data according to the schema.

```python
# Convert results to a DataFrame
df = extractor.to_dataframe(extraction_results)
print(df)

# Display valid results or handle errors
if extraction_results.is_valid:
    for result in extraction_results.get_valid_results():
        print("Extracted Data:", result)
else:
    print("Extraction errors occurred:", extraction_results.validation_errors)

```
## Detailed Workflow

1. **OCR Processing**: Extracts raw text from images using the `OCRProcessor`. This component utilizes OCR to convert image-based content into text.
2. **Schema Selection**: Choose from pre-defined schemas (like Invoice, Receipt, Passport) or create custom schemas to define the structure of data to be extracted.
3. **Data Extraction**: Using the selected schema, the LLM processes the extracted text and returns it as structured data according to the specified fields.
4. **Validation**: Each field undergoes validation based on rules defined in the schema, ensuring accuracy by checking constraints like minimum length, numeric range, and specific patterns.
5. **Output Formats**: Extracted results can be converted into multiple formats, including JSON, pandas DataFrames, or raw dictionaries, making further data processing seamless.

## Core Components for Image Extraction

### `OCRProcessor`

The `OCRProcessor` handles the conversion of image content to text through OCR. It supports multiple OCR models, such as PaddleOCR, which is well-suited for various languages and image resolutions. This component enables easy integration of text extraction from image files.

### `Schema`

Schemas define the structure of data to be extracted from the text, including fields and validation rules. Some of the default schemas include:

- **Invoice**: Extracts fields such as Invoice Number, Date, Customer Name, and Total Amount.
- **Receipt**: Extracts fields such as Receipt Number, Date, Vendor Name, and Payment Method.
- **Passport**: Extracts fields such as Passport Number, Name, Nationality, and Date of Birth.

Each schema specifies fields, data types, and validation rules, allowing flexibility in data extraction.

### `Extractor`

The `Extractor` is the main class responsible for interacting with the LLM, validating extracted data, and formatting output. It integrates with `AsyncIndoxApi` or other LLM classes to process text extracted from images and return structured data according to the specified schema.

### Validation Rules

Validation rules ensure data quality by setting constraints on each field within a schema. Common rules include:

- **Minimum and Maximum Values**: For numeric fields, ensuring values fall within a specified range.
- **String Length Constraints**: For text fields, defining minimum and maximum character limits.
- **Regex Patterns**: To validate specific formats, such as dates or identifiers, ensuring data consistency and reliability.

## Advanced Configuration

### Custom Schema Definition

Create a schema tailored to your extraction needs.

```python
from indoxminer import Field, FieldType, ValidationRule, ExtractorSchema

custom_schema = ExtractorSchema(
    fields=[
        Field(name="transaction_id", description="Unique transaction ID", field_type=FieldType.STRING, required=True),
        Field(name="amount", description="Transaction amount in USD", field_type=FieldType.FLOAT, rules=ValidationRule(min_value=0)),
        Field(name="date", description="Transaction date in YYYY-MM-DD", field_type=FieldType.DATE)
    ]
)


```
## Error Handling and Logging

IndoxMiner provides detailed logging with loguru, which tracks each step of the extraction, including any validation errors. Logs are invaluable for debugging and improving extraction accuracy.


```python
# Configure logging
from loguru import logger

logger.add("extraction.log", level="INFO")


```

## Supported Output Formats

- **JSON**: Returns structured data in JSON format, suitable for further processing or storage.
- **DataFrame**: Converts the results to a pandas DataFrame for analysis and manipulation.
- **Dictionary**: Access the raw extraction results as dictionaries for flexible handling.

## Support and Contributions

For issues, feature requests, or contributions, please submit them via the GitHub issue tracker or make a pull request. Community involvement is highly valued and encouraged!

## License

This project is licensed under the MIT License.


