"""Provides format options.

Adapted from here:
https://stackoverflow.com/a/287944/5755604
"""


class Colors:
    """Definition of formatters.

    Includes both color and font styles.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    OKCYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    def disable(self) -> None:
        """Disable formatter.

        Resets colors and font style.
        """
        self.HEADER = ""
        self.OKBLUE = ""
        self.OKGREEN = ""
        self.OKCYAN = ""
        self.WARNING = ""
        self.FAIL = ""
        self.ENDC = ""
        self.BOLD = ""
        self.UNDERLINE = ""
