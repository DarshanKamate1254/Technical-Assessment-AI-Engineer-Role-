"""Database package for SQLite persistence and query operations."""

from app.database.connection import (
    DEFAULT_DB_PATH,
    get_connection,
    get_db_connection,
)
from app.database.loader import (
    build_database,
    build_database_from_csv,
    prepare_records_for_sqlite,
    store_tickets,
)
from app.database.repository import (
    ALLOWED_COLUMNS,
    ALLOWED_OPERATORS,
    ALLOWED_SORT_COLUMNS,
    TicketRepository,
    get_average_customer_rating_by_agent,
    get_average_resolution_time,
    get_average_resolution_time_by_priority,
    get_average_response_time,
    get_critical_unresolved_count,
    get_high_priority_unresolved_over_24h,
    get_lowest_average_rating_agent,
    get_ticket_by_id,
    get_ticket_count,
    get_ticket_counts_by_category,
    get_ticket_counts_by_priority,
    get_ticket_counts_by_status,
    get_tickets_by_agent,
    get_tickets_by_priority,
    get_tickets_by_status,
    get_unresolved_over_24h_count,
    get_unresolved_ticket_count,
    query_tickets,
)
from app.database.schema import (
    ALL_COLUMNS,
    INDEXES_DDL,
    SCHEMA_DDL,
    TABLE_NAME,
    initialize_database,
)

__all__ = [
    # Connection
    "DEFAULT_DB_PATH",
    "get_connection",
    "get_db_connection",
    # Schema
    "TABLE_NAME",
    "SCHEMA_DDL",
    "INDEXES_DDL",
    "ALL_COLUMNS",
    "initialize_database",
    # Loader
    "store_tickets",
    "build_database",
    "build_database_from_csv",
    "prepare_records_for_sqlite",
    # Repository & Queries
    "TicketRepository",
    "ALLOWED_COLUMNS",
    "ALLOWED_SORT_COLUMNS",
    "ALLOWED_OPERATORS",
    "get_critical_unresolved_count",
    "get_lowest_average_rating_agent",
    "get_unresolved_over_24h_count",
    "get_high_priority_unresolved_over_24h",
    "get_average_resolution_time_by_priority",
    "get_ticket_counts_by_category",
    "get_ticket_counts_by_status",
    "get_average_customer_rating_by_agent",
    "get_ticket_count",
    "get_unresolved_ticket_count",
    "get_ticket_counts_by_priority",
    "get_average_resolution_time",
    "get_average_response_time",
    "get_tickets_by_agent",
    "get_tickets_by_priority",
    "get_tickets_by_status",
    "get_ticket_by_id",
    "query_tickets",
]
