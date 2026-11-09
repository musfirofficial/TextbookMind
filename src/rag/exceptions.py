class IngestionError(Exception):
    """Base exception for all textbook ingestion and parsing errors."""
    pass


class ConfigError(IngestionError):
    """Raised when required configuration or API keys are missing or invalid."""
    pass


class QuotaExceededError(IngestionError):
    """Raised when an external API quota or rate limit is reached (HTTP 429)."""
    pass


class VisionAPIError(IngestionError):
    """Raised when vision multimodal extraction fails."""
    pass
