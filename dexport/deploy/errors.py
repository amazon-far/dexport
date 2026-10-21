"""The error of a deploy session that refuses to run."""


class DeployError(Exception):
    """An inconsistent option, an unusable checkpoint or recording, or inference too slow for the
    loop, found before the loop commands the arms; ``python -m dexport.deploy`` prints it and
    exits."""
