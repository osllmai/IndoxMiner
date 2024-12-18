# IndoxMiner

[![PyPI version](https://badge.fury.io/py/indoxminer.svg)](https://badge.fury.io/py/indoxminer)  
[![License: MIT](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

IndoxMiner is a versatile Python library designed to simplify information extraction from unstructured data sources. It features two core modules:  

1. **Data Extraction**: Extract structured information from text, PDFs, and images using schema-based extraction.  
2. **Object Detection**: Perform object detection on images using a wide range of pre-trained models.  

This library is ideal for automating workflows that require both document processing and image-based analysis.

---

## 🚀 Key Features

### 1. **Data Extraction**
- **Multi-Format Support**: Extract data from text, PDFs, images, and scanned documents.
- **Schema-Based Extraction**: Define custom schemas to specify exactly what data to extract.
- **LLM Integration**: Seamless integration with OpenAI models for intelligent data extraction.
- **Validation & Type Safety**: Built-in validation rules and type-safe field definitions.
- **Flexible Output**: Export data to JSON, pandas DataFrames, or custom formats.
- **OCR Integration**: Multiple OCR engines for image-based text extraction.

### 2. **Object Detection**
- **Model Variety**: Support for a range of state-of-the-art object detection models, including YOLO and Vision Transformers.
- **Ease of Use**: Simple APIs for loading models and running detections.
- **High Customizability**: Configure input queries, models, and output formats.
- **GPU-Optimized**: Run on CUDA-enabled devices for fast inference.

---

## 📦 Installation

```bash
pip install indoxminer
```

---

## 🎯 Quick Start

### Part 1: **Data Extraction**

#### Example: Extracting Text Data

```python
from indoxminer import ExtractorSchema, Field, FieldType, ValidationRule, Extractor, OpenAi

# Initialize OpenAI extractor
llm_extractor = OpenAi(api_key="your-api-key", model="gpt-4-mini")

# Define schema
schema = ExtractorSchema(
    fields=[
        Field(name="product_name", description="Product name", field_type=FieldType.STRING),
        Field(name="price", description="Price in USD", field_type=FieldType.FLOAT)
    ]
)

# Create extractor and process text
extractor = Extractor(llm=llm_extractor, schema=schema)
text = """
MacBook Pro 16-inch with M2 chip
Price: $2,399.99
"""
result = await extractor.extract(text)
print(result.to_dict())
```

#### Example: PDF Processing with Schema Extraction

```python
from indoxminer import DocumentProcessor, ProcessingConfig

processor = DocumentProcessor(
    files=["invoice.pdf"],
    config=ProcessingConfig(hi_res_pdf=True, chunk_size=1000)
)

# Extract data using schema
schema = ExtractorSchema(
    fields=[
        Field(name="bill_to", description="Billing address", field_type=FieldType.STRING),
        Field(name="invoice_date", description="Invoice date", field_type=FieldType.DATE),
        Field(name="total_amount", description="Total amount in USD", field_type=FieldType.FLOAT)
    ]
)
documents = processor.process()
result = await extractor.extract(documents)
```

---

### Part 2: **Object Detection**

IndoxMiner integrates various state-of-the-art pre-trained models for object detection, including YOLO, DETR, and Vision Transformers.

#### Supported Models
- **Transformer-Based Models**: `rtdetr`, `detr`, `detr-clip`
- **YOLO Series**: `yolox`, `yolov5`, `yolov6`, `yolov7`, `yolov8`, `yolov10`, `yolov11`
- **Other Models**: `groundingdino`, `kosmos`, `owlvit`, `detectron2`, `sam2`, `llavanext`

#### Example Usage

```python
from indoxminer.object_detection import IndoxObjectDetection

# Specify model and image path
model_name = "yolov7"
image_path = "/path/to/image.jpg"

# Initialize the detector
indox_detector = IndoxObjectDetection(model_name=model_name, device="cuda")

# Run object detection
indox_detector.run(image_path)
```

---

## 🔧 Configuration Options

### **ProcessingConfig** (For Data Extraction)

```python
config = ProcessingConfig(
    hi_res_pdf=True,          # High-resolution PDF processing
    ocr_enabled=True,         # Enable OCR
    ocr_engine="tesseract",   # OCR engine selection
    chunk_size=1000,          # Text chunk size
    language="en",            # Language for processing
    max_threads=4             # Parallel processing threads
)
```

### **Object Detection Models**
- **Model Name**: Pass the desired model name (e.g., `yolov7`, `detr`) to `IndoxObjectDetection`.
- **Device**: Use `cuda` for GPU acceleration or `cpu` for CPU-only inference.
- **Input Image**: Provide the path to the input image.

---

## 🤝 Contributing

We welcome contributions!  
To contribute:  

1. Fork the repository.  
2. Create a feature branch.  
3. Commit your changes.  
4. Push to the branch.  
5. Open a Pull Request.  

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
