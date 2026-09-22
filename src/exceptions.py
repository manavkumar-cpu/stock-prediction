class StockPredictionError(Exception):
    """Base exception for project-specific failures."""


class ConfigurationError(StockPredictionError):
    """Configuration is missing, invalid, or unsupported."""


class DataDownloadError(StockPredictionError):
    """A data download failed or returned no usable data."""


class DataValidationError(StockPredictionError):
    """Data failed a required validation check."""