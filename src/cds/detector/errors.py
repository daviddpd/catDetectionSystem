class BackendUnavailable(RuntimeError):
    """Raised when a requested backend cannot run in this environment."""


class ModelLoadError(RuntimeError):
    """Raised when model files are missing or unsupported."""
