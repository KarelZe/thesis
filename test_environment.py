"""
Tests if the environment is suited for development.

Raises:
        ValueError: Unrecognized python interpreter.
        TypeError: Python version does not match required version.
"""
import sys

REQUIRED_PYTHON = "python3"


def main() -> None:
    """
    Test, if the development environment is suited.

    Raises:
        ValueError: python interpreter can not be found.
        TypeError: python version can not be found.
    """
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

    if system_major != required_major:
        raise TypeError(
            f"This project requires Python {required_major}. Found: {sys.version}"
        )
    else:
        print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()