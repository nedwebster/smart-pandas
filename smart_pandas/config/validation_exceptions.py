"""Custom validation exceptions for Smart Pandas configurations."""

from typing import List, Optional, Dict, Any


class ConfigValidationError(ValueError):
    """Base exception for configuration validation errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with context information."""
        formatted = f"Configuration Validation Error: {self.message}"
        if self.context:
            formatted += f"\nContext: {self._format_context()}"
        return formatted
    
    def _format_context(self) -> str:
        """Format context information for display."""
        context_lines = []
        for key, value in self.context.items():
            context_lines.append(f"  {key}: {value}")
        return "\n" + "\n".join(context_lines)


class EmptyColumnSetError(ConfigValidationError):
    """Raised when a column set is empty."""
    
    def __init__(self):
        context = {}
        message = "Column set is empty"
        super().__init__(message, context)


class TagCompatibilityError(ConfigValidationError):
    """Raised when incompatible tags are used together."""
    
    def __init__(self, incompatible_tags: List[str], column_name: str):
        self.incompatible_tags = incompatible_tags
        context = {
            "incompatible_tags": incompatible_tags,
            "column_name": column_name,
            "suggestion": "Remove one of the incompatible tags or use separate columns"
        }
        message = "Incompatible tags found"
        super().__init__(message, context)


class TagLimitExceededError(ConfigValidationError):
    """Raised when tag usage exceeds dataset limits."""
    
    def __init__(self, tag_name: str, limit: int, actual: int):
        context = {
            "tag_name": tag_name,
            "limit": limit,
            "actual_count": actual,
            "suggestion": f"Reduce usage of '{tag_name}' tag to at most {limit} column(s)"
        }
        message = "Tag exceeded limit"
        super().__init__(message, context)


class DuplicateColumnError(ConfigValidationError):
    """Raised when duplicate column names are found."""
    
    def __init__(self, duplicate_names: List[str]):
        context = {
            "duplicate_names": duplicate_names,
            "suggestion": "Ensure all column names are unique in the configuration"
        }
        message = "Duplicate column names found"
        super().__init__(message, context)
