from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExtractionResult:
    """Container for single extraction result with validation.

    Attributes:
        data (Dict[str, Any]): Extracted data
        raw_response (str): Original LLM response
        validation_errors (List[str]): List of validation errors
    """
    data: Dict[str, Any]
    raw_response: str
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if extraction passed validation.

        Returns:
            bool: True if no validation errors, False otherwise
        """
        return len(self.validation_errors) == 0


@dataclass
class ExtractionResults:
    """Container for multiple extraction results with validation.

    Attributes:
        combined_data (List[Dict[str, Any]]): List of extracted data
        raw_responses (List[str]): Original LLM responses
        validation_errors (Dict[int, List[str]]): Validation errors by index
    """
    combined_data: List[Dict[str, Any]]
    raw_responses: List[str]
    validation_errors: Dict[int, List[str]]

    @property
    def is_valid(self) -> bool:
        """Check if all extractions passed validation.

        Returns:
            bool: True if no validation errors across all results
        """
        return all(not errors for errors in self.validation_errors.values())

    def get_valid_results(self) -> List[Dict[str, Any]]:
        """Get list of results that passed validation.

        Returns:
            List[Dict[str, Any]]: Valid extraction results
        """
        return [data for i, data in enumerate(self.combined_data)
                if not self.validation_errors.get(i, [])]
