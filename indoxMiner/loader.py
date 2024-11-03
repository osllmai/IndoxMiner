from typing import List, Optional, Union, Dict
from pathlib import Path
import importlib
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from unstructured.documents.elements import Element
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

from .ocr_processor import OCRProcessor


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
class Document:
    """
    A dataclass representing a document with its content and metadata.

    Attributes:
        page_content (str): The textual content of the document page
        metadata (dict): Associated metadata like filename, page number, etc.
    """
    page_content: str
    metadata: dict


@dataclass
class ProcessingConfig:
    """
    Configuration settings for document processing.

    Attributes:
        chunk_size (int): Maximum size of text chunks (default: 500)
        hi_res_pdf (bool): Whether to use high-resolution PDF processing (default: True)
        infer_tables (bool): Whether to detect and process tables (default: False)
        custom_splitter (callable): Custom function for splitting text (default: None)
        max_workers (int): Maximum number of concurrent processing threads (default: 4)
        remove_headers (bool): Whether to remove header elements (default: False)
        remove_references (bool): Whether to remove reference sections (default: False)
        filter_empty_elements (bool): Whether to remove empty elements (default: True)
        ocr_for_images (bool): Whether to perform OCR on images (default: False)
        ocr_model (str): OCR model to use ('tesseract' or 'paddle') (default: 'tesseract')
    """
    chunk_size: int = 500
    hi_res_pdf: bool = True
    infer_tables: bool = False
    custom_splitter: Optional[callable] = None
    max_workers: int = 4
    remove_headers: bool = False
    remove_references: bool = False
    filter_empty_elements: bool = True
    ocr_for_images: bool = False
    ocr_model: str = 'tesseract'


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

        extension = Path(file_path).suffix.lower().lstrip('.')
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
        self.sources = [str(sources)] if isinstance(sources, (str, Path)) else [str(s) for s in sources]
        self.doc_types = {source: DocumentType.from_file(source) for source in self.sources}
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
            'filename': Path(file_path).name,
            'file_directory': str(Path(file_path).parent),
            'filetype': self._get_filetype(file_path),
            'page_number': 1,
            'text_as_html': text,
            'last_modified': datetime.datetime.now().isoformat(),
        }

        element = Text(text=text)
        element.metadata = metadata
        return [element]

    def _filter_elements(self, elements: List[Element]) -> List[Element]:
        """
        Filter elements based on configuration settings.

        Args:
            elements (List[Element]): List of elements to filter

        Returns:
            List[Element]: Filtered list of elements
        """
        if not elements:
            return elements

        filtered = elements

        if self.config.filter_empty_elements:
            filtered = [el for el in filtered if hasattr(el, 'text') and el.text and el.text.strip()]

        if self.config.remove_headers:
            filtered = [el for el in filtered if getattr(el, 'category', '') != "Header"]

        if self.config.remove_references:
            try:
                reference_titles = [
                    el for el in filtered
                    if el.text and el.text.strip().lower() == "references" and getattr(el, 'category', '') == "Title"
                ]
                if reference_titles:
                    reference_id = reference_titles[0].id
                    filtered = [el for el in filtered if getattr(el.metadata, 'parent_id', None) != reference_id]
            except Exception as e:
                print(f"Warning: Could not process references: {e}")

        return filtered

    def _get_elements(self, file_path: str) -> List[Element]:
        """
        Extract elements from a document using appropriate partition function.

        Args:
            file_path (str): Path to the document to process

        Returns:
            List[Element]: Extracted elements from the document
        """
        try:
            if (file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")) and
                    self.config.ocr_for_images):
                text = self.ocr_processor.extract_text(file_path)
                return self._create_element_from_ocr(text, file_path)

            elif file_path.lower().endswith(".pdf"):
                from unstructured.partition.pdf import partition_pdf
                return partition_pdf(
                    filename=file_path,
                    strategy="hi_res" if self.config.hi_res_pdf else "fast",
                    infer_table_structure=self.config.infer_tables,
                )

            elif file_path.lower().endswith((".xlsx", ".xls")):
                from unstructured.partition.xlsx import partition_xlsx
                elements = partition_xlsx(filename=file_path)
                return [el for el in elements if getattr(el.metadata, 'text_as_html', None) is not None]

            elif file_path.lower().startswith(("www", "http")) or file_path.lower().endswith(".html"):
                from unstructured.partition.html import partition_html
                url = file_path if urlparse(file_path).scheme else f"https://{file_path}"
                return partition_html(url=url)

            elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")):
                from unstructured.partition.image import partition_image
                return partition_image(filename=file_path,strategy="auto")

            elif file_path.lower().endswith((".eml", ".msg")):
                from unstructured.partition.email import partition_email
                return partition_email(filename=file_path)

            elif file_path.lower().endswith((".docx", ".doc", ".pptx", ".ppt")):
                content_type = "docx" if file_path.lower().endswith((".docx", ".doc")) else "pptx"
                partition_func = import_unstructured_partition(content_type)
                return partition_func(filename=file_path)

            else:
                doc_type = file_path.lower().split(".")[-1]
                partition_func = import_unstructured_partition(doc_type)
                return partition_func(filename=file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _combine_elements_by_page(self, elements: List[Element]) -> List[Document]:
        """
        Combine elements on the same page into single documents.

        Args:
            elements (List[Element]): Elements to combine

        Returns:
            List[Document]: List of combined page documents
        """
        documents = []

        def get_page_number(element):
            return getattr(element.metadata, 'page_number', 1)

        sorted_elements = sorted(elements, key=get_page_number)

        for page_num, page_elements in groupby(sorted_elements, key=get_page_number):
            page_content = " ".join(el.text for el in page_elements if hasattr(el, 'text') and el.text)
            page_content = page_content.replace("\n", " ").strip()

            if page_content:
                documents.append(page_content)

        return documents

    def _should_chunk_content(self, content: str, chunk_size: int) -> bool:
        """
        Determine if content needs to be chunked based on size.

        Args:
            content (str): Content to evaluate
            chunk_size (int): Maximum chunk size

        Returns:
            bool: True if content should be chunked
        """
        return len(content.split()) > chunk_size

    def _chunk_content(self, content: str, chunk_size: int) -> List[str]:
        """
        Split content into smaller chunks.

        Args:
            content (str): Content to chunk
            chunk_size (int): Maximum size of each chunk

        Returns:
            List[str]: List of content chunks
        """
        if self.config.custom_splitter:
            return self.config.custom_splitter(text=content, max_tokens=chunk_size)

        words = content.split()
        chunks = []
        current_chunk = []
        current_count = 0

        for word in words:
            if current_count + len(word.split()) > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_count = len(word.split())
            else:
                current_chunk.append(word)
                current_count += len(word.split())

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _process_elements_to_document(self, elements: List[Element], source: str) -> List[Document]:
        """
        Convert elements to Document objects with appropriate metadata.

        Args:
            elements (List[Element]): Elements to process
            source (str): Source file path

        Returns:
            List[Document]: Processed document objects
        """
        page_contents = self._combine_elements_by_page(elements)
        documents = []

        for idx, content in enumerate(page_contents, 1):
            if self._should_chunk_content(content, self.config.chunk_size):
                chunks = self._chunk_content(content, self.config.chunk_size)
                for chunk_idx, chunk in enumerate(chunks, 1):
                    metadata = {
                        'filename': Path(source).name,
                        'filetype': self._get_filetype(source),
                        'page_number': idx,
                        'chunk_number': chunk_idx,
                        'source': source
                    }
                    documents.append(Document(page_content=chunk, metadata=metadata))
            else:
                metadata = {
                    'filename': Path(source).name,
                    'filetype': self._get_filetype(source),
                    'page_number': idx,
                    'source': source
                }
                documents.append(Document(page_content=content, metadata=metadata))

        return documents

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

    def process(self, config: Optional[ProcessingConfig] = None) -> Dict[str, List[Document]]:
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
