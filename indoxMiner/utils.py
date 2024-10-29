from typing import Any, Dict, Literal, Tuple, List, Type
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


class Document:
    """Class for storing a piece of text and associated metadata."""

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Initialize the Document with page content and arbitrary metadata."""
        self.page_content = page_content
        self.metadata: Dict[str, Any] = kwargs
        self.type: Literal["Document"] = "Document"

    def __repr__(self) -> str:
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"


def convert_latex_to_md(latex_path):
    """Converts a LaTeX file to Markdown using the latex2markdown library.

    Args:
        latex_path (str): The path to the LaTeX file.

    Returns:
        str: The converted Markdown content, or None if there's an error.
    """
    import latex2markdown
    try:
        with open(latex_path, 'r') as f:
            latex_content = f.read()
            l2m = latex2markdown.LaTeX2Markdown(latex_content)
            markdown_content = l2m.to_markdown()
        return markdown_content
    except FileNotFoundError:
        logger.info(f"Error: LaTeX file not found at {latex_path}")
        return None
    except Exception as e:
        logger.error(f"Error during conversion: {e}")


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


