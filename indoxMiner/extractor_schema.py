from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json
from .schema import Field, OutputFormat


@dataclass
class ExtractorSchema:
    """Schema definition for data extraction with validation and formatting.

    Attributes:
        fields (List[Field]): List of fields to extract
        output_format (OutputFormat): Desired output format
        examples (List[Dict[str, Any]], optional): Example extractions
        context (str, optional): Additional context for extraction
    """
    fields: List[Field]
    output_format: OutputFormat = OutputFormat.JSON
    examples: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None

    def to_prompt(self, text: str) -> str:
        """Generate extraction prompt based on schema.

        Args:
            text (str): Source text for extraction

        Returns:
            str: Formatted prompt for LLM extraction
        """
        fields_desc = "\n".join(f"- {field.to_prompt_string()}"
                                for field in self.fields)

        context_section = f"\nContext:\n{self.context}\n" if self.context else ""

        examples_section = ""
        if self.examples:
            examples_json = json.dumps(self.examples, indent=2)
            examples_section = f"\nExamples:\n{examples_json}\n"

        format_instructions = {
            OutputFormat.JSON: "Format as a JSON object. Use null for missing values.",
            OutputFormat.CSV: "Format as CSV. Use empty string for missing values.",
            OutputFormat.TABLE: "Format as a markdown table. Use empty cells for missing values.",
            OutputFormat.MARKDOWN: "Format as markdown. Use appropriate markdown syntax."
        }
        return f"""Task: Extract structured information from the given text according to the following schema.

        Fields to extract:
        {fields_desc}{context_section}{examples_section}

        Output Requirements:
        1. Extract ONLY the specified fields
        2. Follow the exact field names provided
        3. Use {self.output_format.value} format
        4. {format_instructions[self.output_format]}
        5. If a required field cannot be found, use null/empty values
        6. Validate all values against provided rules
        7. For dates, use ISO format (YYYY-MM-DD)
        8. For lists, provide values in a consistent format
        9. CRITICAL: Return ONLY the {self.output_format.value} output - no explanations, comments, or additional text before or after
        10. CRITICAL: Do not include explanation of what was extracted
        11. CRITICAL: Do not include ```{self.output_format.value} tags or backticks

        Text to analyze:
        {text}

        Return the pure {self.output_format.value} output now:"""

class Schema:
    Passport = ExtractorSchema(
        fields=[
            Field(
                name="Passport Number",
                description="Unique passport number",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^[A-Z0-9]{6,9}$")  # Regex for typical passport format
            ),
            Field(
                name="Name",
                description="Full name of the passport holder",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=3, max_length=50)
            ),
            Field(
                name="Date of Birth",
                description="Date of birth in YYYY-MM-DD format",
                field_type=FieldType.DATE,
                required=True,
            ),
            Field(
                name="Nationality",
                description="Country of nationality",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=2, max_length=30)
            ),
            Field(
                name="Date of Issue",
                description="Issue date of the passport",
                field_type=FieldType.DATE
            ),
            Field(
                name="Date of Expiry",
                description="Expiry date of the passport",
                field_type=FieldType.DATE
            )
        ]
    )

    Invoice = ExtractorSchema(
        fields=[
            Field(
                name="Invoice Number",
                description="Unique number of the invoice",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^\d{4,10}$")  # Sample pattern for invoice number
            ),
            Field(
                name="Date",
                description="Date of the invoice",
                field_type=FieldType.DATE,
                required=True
            ),
            Field(
                name="Customer Name",
                description="Name of the customer",
                field_type=FieldType.STRING,
                rules=ValidationRule(min_length=3, max_length=50)
            ),
            Field(
                name="Total Amount",
                description="Total amount in the invoice",
                field_type=FieldType.FLOAT,
                required=True,
                rules=ValidationRule(min_value=0.0)
            ),
            Field(
                name="Item List",
                description="List of items in the invoice",
                field_type=FieldType.LIST,
                array_item_type=FieldType.STRING
            )
        ]
    )

    Receipt = ExtractorSchema(
        fields=[
            Field(
                name="Receipt Number",
                description="Unique receipt number",
                field_type=FieldType.STRING,
                rules=ValidationRule(pattern=r"^\d{5,10}$")
            ),
            Field(
                name="Date",
                description="Date of the receipt",
                field_type=FieldType.DATE,
                required=True
            ),
            Field(
                name="Vendor Name",
                description="Name of the vendor or store",
                field_type=FieldType.STRING,
                rules=ValidationRule(min_length=2, max_length=50)
            ),
            Field(
                name="Total Amount",
                description="Total amount paid",
                field_type=FieldType.FLOAT,
                required=True,
                rules=ValidationRule(min_value=0.0)
            ),
            Field(
                name="Payment Method",
                description="Method of payment used (e.g., cash, card)",
                field_type=FieldType.STRING,
                rules=ValidationRule(allowed_values=["cash", "card", "online"])
            )
        ]
    )

    ID_Card = ExtractorSchema(
        fields=[
            Field(
                name="ID Number",
                description="Identification number on the card",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^\d{5,12}$")
            ),
            Field(
                name="Full Name",
                description="Full name of the cardholder",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=3, max_length=50)
            ),
            Field(
                name="Date of Birth",
                description="Date of birth in YYYY-MM-DD format",
                field_type=FieldType.DATE,
                required=True
            ),
            Field(
                name="Nationality",
                description="Nationality of the cardholder",
                field_type=FieldType.STRING,
                rules=ValidationRule(min_length=2, max_length=30)
            ),
            Field(
                name="Gender",
                description="Gender of the cardholder",
                field_type=FieldType.STRING,
                rules=ValidationRule(allowed_values=["Male", "Female", "Other"])
            ),
            Field(
                name="Date of Issue",
                description="Issue date of the ID card",
                field_type=FieldType.DATE
            ),
            Field(
                name="Date of Expiry",
                description="Expiry date of the ID card",
                field_type=FieldType.DATE
            )
        ]
    )

    Bank_Statement = ExtractorSchema(
        fields=[
            Field(
                name="Account Holder Name",
                description="Name of the account holder",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=3, max_length=50)
            ),
            Field(
                name="Account Number",
                description="Unique bank account number",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^\d{10,12}$")
            ),
            Field(
                name="Transaction Date",
                description="Date of the transaction",
                field_type=FieldType.DATE,
                required=True
            ),
            Field(
                name="Transaction Description",
                description="Description of the transaction",
                field_type=FieldType.STRING
            ),
            Field(
                name="Transaction Amount",
                description="Amount involved in the transaction",
                field_type=FieldType.FLOAT,
                required=True,
                rules=ValidationRule(min_value=0.0)
            ),
            Field(
                name="Balance After Transaction",
                description="Account balance after the transaction",
                field_type=FieldType.FLOAT
            )
        ]
    )

    Medical_Record = ExtractorSchema(
        fields=[
            Field(
                name="Patient Name",
                description="Full name of the patient",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=3, max_length=50)
            ),
            Field(
                name="Date of Birth",
                description="Date of birth in YYYY-MM-DD format",
                field_type=FieldType.DATE,
                required=True
            ),
            Field(
                name="Medical Record Number",
                description="Unique ID for the medical record",
                field_type=FieldType.STRING
            ),
            Field(
                name="Diagnosis",
                description="Medical diagnosis of the patient",
                field_type=FieldType.STRING,
                required=True
            ),
            Field(
                name="Medications",
                description="List of medications prescribed",
                field_type=FieldType.LIST,
                array_item_type=FieldType.STRING
            ),
            Field(
                name="Treatment Dates",
                description="Dates when treatments were given",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DATE
            ),
            Field(
                name="Physician Name",
                description="Name of the attending physician",
                field_type=FieldType.STRING
            )
        ]
    )