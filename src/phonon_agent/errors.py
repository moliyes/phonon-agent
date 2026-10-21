class PhononAgentError(RuntimeError):
    """Base error for user-facing workflow failures."""


class ConfigError(PhononAgentError):
    pass


class WaitingForUser(PhononAgentError):
    """A scientifically meaningful decision or missing input blocks progress."""


class ExternalProgramError(PhononAgentError):
    pass
