"""
Tests for the classical classifier.

Use of artificial data to test the classifier.
"""
import unittest
import pandas as pd

from otc.models.classical_classifier import ClassicalClassifier


class TestClassicalClassifier(unittest.TestCase):
    """
    Perform automated tests for ClassicalClassifier.

    Args:
        unittest (_type_): unittest module
    """

    def setUp(self) -> None:
        """
        Set up basic classifier and data.

        Prepares inputs and expected outputs for testing.
        """
        self.x_train = pd.DataFrame(
            [[1, 2], [3, 4], [1, 2], [3, 4]], columns=["BEST_ASK", "BEST_BID"]
        )
        self.y_train = pd.Series([1, 1, -1, -1])
        self.x_test = pd.DataFrame(
            [[1, 2], [3, 4], [1, 2], [3, 4]], columns=["BEST_ASK", "BEST_BID"]
        )
        self.y_test = pd.Series([1, 1, -1, -1])
        self.random_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], random_state=42
        ).fit(self.x_train, self.y_train)

    def test_shapes(self) -> None:
        """
        Test, if shapes of the classifier equal the targets.
        """
        y_pred = self.random_classifier.predict(self.x_test)

        self.assertEqual(self.y_test.shape, y_pred.shape)

    def test_random_state(self) -> None:
        """
        Test, if random state is correctly set.

        Two classifiers with the same random state should give the same results.
        """
        first_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], random_state=50
        ).fit(self.x_train, self.y_train)
        first_y_pred = first_classifier.predict(self.x_test)

        second_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], random_state=50
        ).fit(self.x_train, self.y_train)
        second_y_pred = second_classifier.predict(self.x_test)

        self.assertTrue((first_y_pred == second_y_pred).all())

    def test_score(self) -> None:
        """
        Test, if score is correctly calculated..

        For a random classification i. e., `layers=[("nan", "ex")]`, the score 
        should be around 0.5.
        """
        accuracy = self.random_classifier.score(self.x_test, self.y_test)
        self.assertAlmostEqual(accuracy, 0.5, delta=0.25)  # type: ignore

    def test_fit(self) -> None:
        """
        Test, if fit works.

        A fitted classifier should have an attribute `layers_`.
        """
        fitted_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], random_state=42
        ).fit(self.x_train, self.y_train)
        self.assertTrue(hasattr(fitted_classifier, "layers_"))
