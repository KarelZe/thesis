"""
Perform automated tests.
Corresponds to setup.py
"""

import os
import unittest


class TestSetup(unittest.TestCase):
    """
    Perform automated tests.
    Args:
        unittest unittest.TestCase: testcase
    """

    def test_entrypoint(self) -> None:
        """
        Test if script can be called.
        Call with --help optional to prevent execution.
        """
        exit_status = os.system("python setup.py --help")
        assert exit_status == 0
