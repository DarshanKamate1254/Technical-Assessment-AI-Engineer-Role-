"""Pipeline package for data ingestion, parsing, validation, and cleaning."""

from app.pipeline.cleaner import (
    CATEGORY_MAP,
    PRIORITY_MAP,
    STATUS_MAP,
    DataCleaner,
)
from app.pipeline.parser import (
    CSVReadError,
    DATETIME_COLUMNS,
    EXPECTED_COLUMNS,
    MissingColumnError,
    NUMERIC_COLUMNS,
    ParserError,
    ParsingDiagnostics,
    STRING_COLUMNS,
    convert_column_types,
    parse_csv,
    parse_csv_with_diagnostics,
    read_raw_csv,
    validate_columns,
)
from app.pipeline.pipeline import (
    validate_and_clean,
    validate_and_clean_with_report,
)
from app.pipeline.validator import (
    RATING_MAX,
    RATING_MIN,
    REQUIRED_COLUMNS,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    VALID_STATUSES,
    DataValidator,
    SchemaValidationError,
    ValidationError,
    ValidationReport,
    validate_schema,
)

__all__ = [
    # Parser exports
    "parse_csv",
    "parse_csv_with_diagnostics",
    "ParsingDiagnostics",
    "ParserError",
    "CSVReadError",
    "MissingColumnError",
    "read_raw_csv",
    "validate_columns",
    "convert_column_types",
    "EXPECTED_COLUMNS",
    "STRING_COLUMNS",
    "DATETIME_COLUMNS",
    "NUMERIC_COLUMNS",
    # Validator exports
    "DataValidator",
    "ValidationReport",
    "ValidationError",
    "SchemaValidationError",
    "validate_schema",
    "REQUIRED_COLUMNS",
    "VALID_CATEGORIES",
    "VALID_PRIORITIES",
    "VALID_STATUSES",
    "RATING_MIN",
    "RATING_MAX",
    # Cleaner exports
    "DataCleaner",
    "CATEGORY_MAP",
    "PRIORITY_MAP",
    "STATUS_MAP",
    # Pipeline exports
    "validate_and_clean",
    "validate_and_clean_with_report",
]
