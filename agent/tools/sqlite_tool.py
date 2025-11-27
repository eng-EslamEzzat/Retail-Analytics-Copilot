"""SQLite database access and schema introspection tools."""
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SQLiteTool:
    """Tool for interacting with SQLite database."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Get database schema information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema = {}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            # Quote table name if it contains spaces or special characters
            quoted_name = f'"{table_name}"' if ' ' in table_name else table_name
            cursor.execute(f"PRAGMA table_info({quoted_name})")
            columns = cursor.fetchall()
            schema[table_name] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": col[3],
                    "default": col[4],
                    "pk": col[5]
                }
                for col in columns
            ]
        
        conn.close()
        return schema
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    
    def execute_query(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], List[str]]:
        """
        Execute SQL query and return results.
        
        Returns:
            Tuple of (rows as list of dicts, error message if any, column names)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch all rows
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]
            
            conn.close()
            return result, None, columns
            
        except sqlite3.Error as e:
            return None, str(e), []
        except Exception as e:
            return None, f"Unexpected error: {str(e)}", []
    
    def get_schema_string(self) -> str:
        """Get schema as a formatted string for prompts."""
        schema = self.get_schema()
        lines = []
        lines.append("IMPORTANT: Table names with spaces must be quoted with double quotes.")
        lines.append("Example: \"Order Details\" not Order Details")
        lines.append("")
        for table_name, columns in schema.items():
            # Quote table name if it has spaces
            quoted_name = f'"{table_name}"' if ' ' in table_name else table_name
            lines.append(f"\nTable: {table_name} (use {quoted_name} in SQL)")
            for col in columns:
                col_type = col['type']
                pk = " (PRIMARY KEY)" if col['pk'] else ""
                notnull = " NOT NULL" if col['notnull'] else ""
                lines.append(f"  - {col['name']}: {col_type}{pk}{notnull}")
        return "\n".join(lines)

