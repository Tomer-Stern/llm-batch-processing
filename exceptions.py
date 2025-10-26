"""
Custom exceptions for batch processing toolkit.
"""


class BatchProcessingError(Exception):
    """Base exception for batch processing errors."""
    pass


class APIKeyError(BatchProcessingError):
    """Raised when API key is missing or invalid."""
    pass


class BatchSubmissionError(BatchProcessingError):
    """Raised when batch submission fails."""
    pass


class BatchProcessingTimeoutError(BatchProcessingError):
    """Raised when batch processing times out."""
    pass


class InvalidInputError(BatchProcessingError):
    """Raised when input data is invalid."""
    pass


class ConfigurationError(BatchProcessingError):
    """Raised when configuration is invalid."""
    pass


class CostEstimationError(BatchProcessingError):
    """Raised when cost estimation fails."""
    pass


class ResultParsingError(BatchProcessingError):
    """Raised when parsing results fails."""
    pass
