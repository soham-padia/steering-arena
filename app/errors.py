"""Domain errors for the submit pipeline. main.py maps these to HTTP codes."""


class SubmitError(Exception):
    """Base for user-facing submit failures. `message` is safe to show directly."""

    http_status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationError(SubmitError):
    http_status = 400


class DuplicateError(SubmitError):
    http_status = 409

    def __init__(self, message: str, existing: dict, rank: int):
        super().__init__(message)
        self.existing = existing
        self.rank = rank


class RateLimited(SubmitError):
    http_status = 429


class QueueFull(SubmitError):
    http_status = 503


class ScoringUnavailable(SubmitError):
    http_status = 503
