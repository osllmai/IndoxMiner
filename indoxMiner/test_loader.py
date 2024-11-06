from typing import List, Optional, Union, Dict
from pathlib import Path
import importlib
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from unstructured.documents.elements import Element
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
from .utils import Document
from .ocr_processor import OCRProcessor
from typing import Tuple, Type


def filter_complex_metadata(
    documents: List[Document],
    *,
    allowed_types: Tuple[Type, ...] = (str, bool, int, float),
) -> List[Document]:
    """Filter out metadata types that are not supported for a vector store."""
    updated_documents = []
    for document in documents:
        filtered_metadata = {}
        for key, value in document.metadata.items():
            if not isinstance(value, allowed_types):
                continue
            filtered_metadata[key] = value

        document.metadata = filtered_metadata
        updated_documents.append(document)

    return updated_documents


def import_unstructured_partition(content_type):
    """
    Dynamically imports the appropriate partition function from the unstructured library.

    Args:
        content_type (str): The type of content to process (e.g., 'pdf', 'docx')

    Returns:
        callable: The partition function for the specified content type
    """
    module_name = f"unstructured.partition.{content_type}"
    module = importlib.import_module(module_name)
    partition_function_name = f"partition_{content_type}"
    return getattr(module, partition_function_name)


@dataclass
class ProcessingConfig:
    """
    Configuration settings for document processing.

    Attributes:
        chunk_size (int): Maximum size of text chunks (default: 500)
        hi_res_pdf (bool): Whether to use high-resolution PDF processing (default: True)
        infer_tables (bool): Whether to detect and process tables (default: False)
        max_workers (int): Maximum number of concurrent processing threads (default: 4)
        remove_headers (bool): Whether to remove header elements (default: False)
        remove_references (bool): Whether to remove reference sections (default: False)
        filter_empty_elements (bool): Whether to remove empty elements (default: True)
        ocr_for_images (bool): Whether to perform OCR on images (default: False)
        ocr_model (str): OCR model to use ('tesseract' or 'paddle') (default: 'tesseract')
    """

    chunk_size: int = 4048
    hi_res_pdf: bool = True
    infer_tables: bool = False
    max_workers: int = 4
    remove_headers: bool = False
    remove_references: bool = False
    filter_empty_elements: bool = True
    ocr_for_images: bool = False
    ocr_model: str = "tesseract"


class DocumentType(Enum):
    """
    Enumeration of supported document types with their corresponding file extensions.
    """

    BMP = "bmp"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    EML = "eml"
    EPUB = "epub"
    HEIC = "heic"
    HTML = "html"
    JPEG = "jpeg"
    JPG = "jpg"
    MARKDOWN = "md"
    MSG = "msg"
    ODT = "odt"
    ORG = "org"
    P7S = "p7s"
    PDF = "pdf"
    PNG = "png"
    PPT = "ppt"
    PPTX = "pptx"
    RST = "rst"
    RTF = "rtf"
    TIFF = "tiff"
    TEXT = "txt"
    TSV = "tsv"
    XLS = "xls"
    XLSX = "xlsx"
    XML = "xml"

    @classmethod
    def from_file(cls, file_path: str) -> "DocumentType":
        """
        Determines the document type from a file path or URL.

        Args:
            file_path (str): Path or URL to the document

        Returns:
            DocumentType: The determined document type

        Raises:
            ValueError: If the file type is not supported
        """
        if file_path.lower().startswith(("http://", "https://", "www.")):
            return cls.HTML

        extension = Path(file_path).suffix.lower().lstrip(".")
        if extension == "jpg":
            extension = "jpeg"

        try:
            return cls(extension)
        except ValueError:
            raise ValueError(f"Unsupported file type: {extension}")


class DocumentProcessor:
    """
    A processor for extracting and structuring content from various document types.

    This class handles the extraction of text and metadata from different document formats,
    including PDFs, Office documents, images, and web content. It supports concurrent
    processing, content chunking, and various filtering options.

    Attributes:
        sources (List[str]): List of file paths or URLs to process
        doc_types (Dict[str, DocumentType]): Mapping of sources to their document types
        ocr_processor (Optional[OCRProcessor]): Processor for optical character recognition
    """

    def __init__(self, sources: Union[str, Path, List[Union[str, Path]]]):
        """
        Initialize the DocumentProcessor with one or more sources.

        Args:
            sources: Single source or list of sources to process
        """
        self.sources = (
            [str(sources)]
            if isinstance(sources, (str, Path))
            else [str(s) for s in sources]
        )
        self.doc_types = {
            source: DocumentType.from_file(source) for source in self.sources
        }
        self.ocr_processor = None

    def _init_ocr_processor(self):
        """Initialize OCR processor if OCR processing is enabled."""
        if self.config.ocr_for_images and not self.ocr_processor:
            self.ocr_processor = OCRProcessor(model=self.config.ocr_model)

    def _create_element_from_ocr(self, text: str, file_path: str) -> List[Element]:
        """
        Create Element objects from OCR-extracted text.

        Args:
            text (str): Extracted text from OCR
            file_path (str): Path to the processed file

        Returns:
            List[Element]: List containing the created Element object
        """
        from unstructured.documents.elements import Text
        import datetime

        metadata = {
            "filename": Path(file_path).name,
            "file_directory": str(Path(file_path).parent),
            "filetype": self._get_filetype(file_path),
            "page_number": 1,
            "text_as_html": text,
            "last_modified": datetime.datetime.now().isoformat(),
        }

        element = Text(text=text)
        element.metadata = metadata
        return [element]

    def _get_elements(self, file_path: str) -> List[Element]:
        try:
            if (
                file_path.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")
                )
                and self.config.ocr_for_images
            ):
                text = self.ocr_processor.extract_text(file_path)
                return self._create_element_from_ocr(text, file_path)
            elif file_path.lower().endswith(".pdf"):
                # Partition PDF with a high-resolution strategy
                from unstructured.partition.pdf import partition_pdf

                elements = partition_pdf(
                    filename=file_path,
                    strategy="hi_res",
                    # infer_table_structure=True,
                )
                # Remove "References" and header elements
                reference_title = [
                    el
                    for el in elements
                    if el.text == "References" and el.category == "Title"
                ][0]
                references_id = reference_title.id
                elements = [
                    el for el in elements if el.metadata.parent_id != references_id
                ]
                elements = [el for el in elements if el.category != "Header"]
            elif file_path.lower().endswith(".xlsx"):
                from unstructured.partition.xlsx import partition_xlsx

                elements_ = partition_xlsx(filename=file_path)
                elements = [
                    el for el in elements_ if el.metadata.text_as_html is not None
                ]
            elif file_path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")
            ):
                from unstructured.partition.image import partition_image

                elements = partition_image(filename=file_path, strategy="auto")
            elif file_path.lower().startswith("www") or file_path.lower().startswith(
                "http"
            ):
                from unstructured.partition.html import partition_html

                elements = partition_html(url=file_path)
            else:
                if file_path.lower().endswith(".tex"):
                    file_path = convert_latex_to_md(latex_path=file_path)
                content_type = file_path.lower().split(".")[-1]
                if content_type == "txt":
                    prt = import_unstructured_partition(content_type="text")
                else:
                    prt = import_unstructured_partition(content_type=content_type)
                elements = prt(filename=file_path)
            return elements
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    # def _combine_elements_by_page(self, elements: List[Element]) -> List[Document]:

    def _process_elements_to_document(self, file_path):
        from unstructured.chunking.title import by_page

        try:
            elements = self._get_elements(file_path)

            # if splitter:
            #     text = ""
            #     for el in elements:
            #         text += el.text

            #     documents = splitter(text=text, max_tokens=chunk_size)
            # else:
            # Split elements based on the title and the specified max characters per chunk
            elements = chunk_by_title(elements, max_characters=chunk_size)
            documents = []

            # Convert each element into a `Document` object with relevant metadata
            for element in elements:
                metadata = element.metadata.to_dict()
                del metadata["languages"]  # Remove unnecessary metadata field

                for key, value in metadata.items():
                    if isinstance(value, list):
                        value = str(value)
                    metadata[key] = value

                # documents.append(Document(page_content=element.text, metadata=**metadata))
                documents.append(
                    Document(page_content=element.text.replace("\n", ""), **metadata)
                )

                # Filter and sanitize complex metadata
                documents = filter_complex_metadata(documents=documents)

                return documents

        except Exception as e:
            logger.error(f"Failed at step with error: {e}")
            raise

    def _get_filetype(self, source: str) -> str:
        """Get MIME type for the file."""
        doc_type = self.doc_types[source]
        mime_types = {
            DocumentType.PDF: "application/pdf",
            DocumentType.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            DocumentType.XLS: "application/vnd.ms-excel",
            DocumentType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            DocumentType.DOC: "application/msword",
            DocumentType.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            DocumentType.PPT: "application/vnd.ms-powerpoint",
            DocumentType.HTML: "text/html",
            DocumentType.TEXT: "text/plain",
            DocumentType.MARKDOWN: "text/markdown",
            DocumentType.XML: "application/xml",
            DocumentType.CSV: "text/csv",
            DocumentType.TSV: "text/tab-separated-values",
            DocumentType.RTF: "application/rtf",
            DocumentType.EPUB: "application/epub+zip",
            DocumentType.MSG: "application/vnd.ms-outlook",
            DocumentType.EML: "message/rfc822",
            DocumentType.PNG: "image/png",
            DocumentType.JPEG: "image/jpeg",
            DocumentType.TIFF: "image/tiff",
            DocumentType.BMP: "image/bmp",
            DocumentType.HEIC: "image/heic",
        }
        return mime_types.get(doc_type, "application/octet-stream")

    def process(
        self, config: Optional[ProcessingConfig] = None
    ) -> Dict[str, List[Document]]:
        """Process all documents with the given configuration."""
        self.config = config or ProcessingConfig()

        # Initialize OCR processor if needed
        self._init_ocr_processor()

        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_source = {
                executor.submit(self._get_elements, source): source
                for source in self.sources
            }

            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    elements = future.result()
                    filtered_elements = self._filter_elements(elements)
                    results[Path(source).name] = self._process_elements_to_document(
                        filtered_elements, source
                    )
                except Exception as e:
                    print(f"Failed to process {source}: {e}")
                    results[Path(source).name] = []

        return results
