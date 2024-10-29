from typing import List, Optional, Union, Dict
from pathlib import Path
import importlib
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
from unstructured.partition.common import Element
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby

from .ocr_processor import OCRProcessor


def import_unstructured_partition(content_type):
    # Import appropriate partition function from the `unstructured` library
    module_name = f"unstructured.partition.{content_type}"
    module = importlib.import_module(module_name)
    partition_function_name = f"partition_{content_type}"
    prt = getattr(module, partition_function_name)
    return prt


@dataclass
class Document:
    """Document class with page content and metadata."""
    page_content: str
    metadata: dict


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 500
    hi_res_pdf: bool = True
    infer_tables: bool = False
    custom_splitter: Optional[callable] = None
    max_workers: int = 4
    remove_headers: bool = False
    remove_references: bool = False
    filter_empty_elements: bool = True
    ocr_for_images: bool = False
    ocr_model: str = 'tesseract'  # 'tesseract' or 'paddle'


class DocumentType(Enum):
    """Document types supported by the processor."""
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
        """Determine document type from file path."""
        if file_path.lower().startswith(("http://", "https://", "www.")):
            return cls.HTML

        extension = Path(file_path).suffix.lower().lstrip('.')

        # Handle jpg/jpeg special case
        if extension == "jpg":
            extension = "jpeg"

        try:
            return cls(extension)
        except ValueError:
            raise ValueError(f"Unsupported file type: {extension}")


class DocumentProcessor:
    def __init__(self, sources: Union[str, Path, List[Union[str, Path]]]):
        self.sources = [str(sources)] if isinstance(sources, (str, Path)) else [str(s) for s in sources]
        self.doc_types = {source: DocumentType.from_file(source) for source in self.sources}
        self.ocr_processor = None

    def _init_ocr_processor(self):
        """Initialize OCR processor if needed"""
        if self.config.ocr_for_images and not self.ocr_processor:
            self.ocr_processor = OCRProcessor(model=self.config.ocr_model)

    def _create_element_from_ocr(self, text: str, file_path: str) -> List[Element]:
        """Create a proper Element object from OCR text"""
        from unstructured.documents.elements import Text
        import datetime

        # Create metadata
        metadata = {
            'filename': Path(file_path).name,
            'file_directory': str(Path(file_path).parent),
            'filetype': self._get_filetype(file_path),
            'page_number': 1,
            'text_as_html': text,  # Include the text as HTML
            'last_modified': datetime.datetime.now().isoformat(),
        }

        # Create the element with proper metadata
        element = Text(text=text)
        element.metadata = metadata
        return [element]

    def _filter_elements(self, elements: List[Element]) -> List[Element]:
        """Filter elements based on configuration."""
        if not elements:
            return elements

        filtered = elements

        # Filter empty elements if configured
        if self.config.filter_empty_elements:
            filtered = [el for el in filtered if hasattr(el, 'text') and el.text and el.text.strip()]

        # Remove headers if configured
        if self.config.remove_headers:
            filtered = [el for el in filtered if getattr(el, 'category', '') != "Header"]

        # Remove references section if configured
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
        """Get elements from the document using appropriate partition function."""
        try:
            # Handle image files with OCR if configured
            if (file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")) and
                    self.config.ocr_for_images):
                text = self.ocr_processor.extract_text(file_path)
                return self._create_element_from_ocr(text, file_path)

            # Handle PDF files
            elif file_path.lower().endswith(".pdf"):
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(
                    filename=file_path,
                    strategy="hi_res" if self.config.hi_res_pdf else "fast",
                    infer_table_structure=self.config.infer_tables,
                )
                return elements

            # Handle Excel files
            elif file_path.lower().endswith((".xlsx", ".xls")):
                from unstructured.partition.xlsx import partition_xlsx
                elements = partition_xlsx(filename=file_path)
                return [el for el in elements if getattr(el.metadata, 'text_as_html', None) is not None]

            # Handle HTML and web content
            elif file_path.lower().startswith(("www", "http")) or file_path.lower().endswith(".html"):
                from unstructured.partition.html import partition_html
                url = file_path if urlparse(file_path).scheme else f"https://{file_path}"
                return partition_html(url=url)


            # Handle image files
            elif file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heic")):
                from unstructured.partition.image import partition_image
                return partition_image(filename=file_path)

            # Handle email files
            elif file_path.lower().endswith((".eml", ".msg")):
                from unstructured.partition.email import partition_email
                return partition_email(filename=file_path)

            # Handle common office documents
            elif file_path.lower().endswith((".docx", ".doc", ".pptx", ".ppt")):
                content_type = "docx" if file_path.lower().endswith((".docx", ".doc")) else "pptx"
                partition_func = import_unstructured_partition(content_type)
                return partition_func(filename=file_path)

            # Handle all other supported formats
            else:
                doc_type = file_path.lower().split(".")[-1]
                module_name = f"unstructured.partition.{doc_type}"
                try:
                    module = importlib.import_module(module_name)
                    partition_func = getattr(module, f"partition_{doc_type}")
                    return partition_func(filename=file_path)
                except (ImportError, AttributeError):
                    print(f"Unsupported file type: {doc_type}")
                    return []

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def _combine_elements_by_page(self, elements: List[Element]) -> List[Document]:
        """Combine all elements on the same page into a single document."""
        documents = []

        # Group elements by page number
        def get_page_number(element):
            return getattr(element.metadata, 'page_number', 1)

        # Sort elements by page number
        sorted_elements = sorted(elements, key=get_page_number)

        # Group elements by page
        for page_num, page_elements in groupby(sorted_elements, key=get_page_number):
            # Combine all text from elements on this page
            page_content = " ".join(el.text for el in page_elements if hasattr(el, 'text') and el.text)
            page_content = page_content.replace("\n", " ").strip()

            if page_content:  # Only create document if there's content
                documents.append(page_content)

        return documents

    def _should_chunk_content(self, content: str, chunk_size: int) -> bool:
        """Determine if content should be chunked based on size."""
        return len(content.split()) > chunk_size

    def _chunk_content(self, content: str, chunk_size: int) -> List[str]:
        """Chunk content into smaller pieces."""
        if self.config.custom_splitter:
            return self.config.custom_splitter(text=content, max_tokens=chunk_size)

        words = content.split()
        chunks = []
        current_chunk = []
        current_count = 0

        for word in words:
            if current_count + len(word.split()) > chunk_size:
                if current_chunk:  # Only append if there's content
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_count = len(word.split())
            else:
                current_chunk.append(word)
                current_count += len(word.split())

        if current_chunk:  # Add the last chunk if it exists
            chunks.append(" ".join(current_chunk))

        return chunks

    def _process_elements_to_document(self, elements: List[Element], source: str) -> List[Document]:
        """Convert elements to Document objects with combined page content."""
        # First combine elements by page
        page_contents = self._combine_elements_by_page(elements)
        documents = []

        for idx, content in enumerate(page_contents, 1):
            # Check if content needs chunking
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