"""
Tests for the classical classifier.

Use of artificial data to test the classifier.
"""

import numpy as np
import pandas as pd
import pytest

from otc.models.classical_classifier import ClassicalClassifier


class TestClassicalClassifier:
    """
    Perform automated tests for ClassicalClassifier.

    Args:
        unittest (_type_): unittest module
    """

    def setup(self) -> None:
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

        Shapes are usually [no. of samples, 1].
        """
        y_pred = self.random_classifier.predict(self.x_test)

        assert self.y_test.shape == y_pred.shape

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

        assert (first_y_pred == second_y_pred).all()

    def test_score(self) -> None:
        """
        Test, if score is correctly calculated..

        For a random classification i. e., `layers=[("nan", "ex")]`, the score
        should be around 0.5.
        """
        accuracy = self.random_classifier.score(self.x_test, self.y_test)
        assert abs(accuracy - 0.5) <= 0.25  # type: ignore

    def test_fit(self) -> None:
        """
        Test, if fit works.

        A fitted classifier should have an attribute `layers_`.
        """
        fitted_classifier = ClassicalClassifier(
            layers=[("nan", "ex")], random_state=42
        ).fit(self.x_train, self.y_train)
        assert hasattr(fitted_classifier, "layers_")

    def test_invalid_func(self) -> None:
        """
        Test, if only valid function strings can be passed.

        An exception should be raised for invalid function strings.
        Test for 'foo', which is no valid rule.
        """
        classifier = ClassicalClassifier(layers=[("foo", "all")], random_state=42)
        with pytest.raises(ValueError):
            classifier.fit(self.x_train, self.y_train)

    def test_invalid_subset(self) -> None:
        """
        Test, if only valid subset strings can be passed.

        An exception should be raised for invalid subsets.
        Test for 'bar', which is no valid subset.
        """
        classifier = ClassicalClassifier(layers=[("tick", "bar")], random_state=42)
        with pytest.raises(ValueError):
            classifier.fit(self.x_train, self.y_train)

    def test_override(self) -> None:
        """
        Test, if classifier does not override valid results from layer one.

        If all data can be classified using first rule, first rule should
        only be applied.
        """
        x_train = pd.DataFrame(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            columns=["TRADE_PRICE", "price_ex_lag", "price_all_lead"],
        )
        y_train = pd.Series([0, 0, 0])
        x_test = pd.DataFrame(
            [[1, 2, 0], [2, 1, 3]],
            columns=["TRADE_PRICE", "price_ex_lag", "price_all_lead"],
        )
        y_test = pd.Series([-1, 1])

        fitted_classifier = ClassicalClassifier(
            layers=[("tick", "ex"), ("rev_tick", "all")], random_state=7
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["all", "ex"])
    def test_tick_rule(self, subset: str) -> None:
        """
        Test, if tick rule is correctly applied.

        Tests cases where prev. trade price is higher, lower, equal or missing.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0], [0, 0], [0, 0]], columns=["TRADE_PRICE", f"price_{subset}_lag"]
        )
        y_train = pd.Series([0, 0, 0])
        x_test = pd.DataFrame(
            [[1, 2], [2, 1], [1, 1], [1, np.nan]],
            columns=["TRADE_PRICE", f"price_{subset}_lag"],
        )

        # first two by rule (see p. 28 Grauer et al.), remaining two by random chance.
        y_test = pd.Series([-1, 1, 1, -1])

        fitted_classifier = ClassicalClassifier(
            layers=[("tick", subset)], random_state=7
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["all", "ex"])
    def test_rev_tick_rule(self, subset: str) -> None:
        """
        Test, if rev. tick rule is correctly applied.

        Tests cases where suc. trade price is higher, lower, equal or missing.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0], [0, 0], [0, 0]], columns=["TRADE_PRICE", f"price_a{subset}_lead"]
        )
        y_train = pd.Series([0, 0, 0])
        x_test = pd.DataFrame(
            [[1, 2], [2, 1], [1, 1], [1, np.nan]],
            columns=["TRADE_PRICE", f"price_{subset}_lead"],
        )

        # first two by rule (see p. 28 Grauer et al.), remaining two by random chance.
        y_test = pd.Series([-1, 1, 1, -1])

        fitted_classifier = ClassicalClassifier(
            layers=[("rev_tick", subset)], random_state=7
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_quote_rule(self, subset: str) -> None:
        """
        Test, if quote rule is correctly applied.

        Tests cases where prev. trade price is higher, lower, equal or missing.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by rule (see p. 28 Grauer et al.), remaining four by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3],
                [3, 1, 3],
                [1, 1, 1],
                [3, 2, 4],
                [1, np.nan, 1],
                [3, np.nan, np.nan],
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}"],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        fitted_classifier = ClassicalClassifier(
            layers=[("quote", subset)], random_state=45
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        print(y_pred)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_lr(self, subset: str) -> None:
        """
        Test, if the lr algorithm is correctly applied.

        Tests cases where both quote rule and tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by quote rule, remaining two by tick rule.
        x_test = pd.DataFrame(
            [[1, 1, 3, 0], [3, 1, 3, 0], [1, 1, 1, 0], [3, 2, 4, 4]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_test = pd.Series([-1, 1, 1, -1])
        fitted_classifier = ClassicalClassifier(
            layers=[("lr", subset)], random_state=7
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_lr(self, subset: str) -> None:
        """
        Test, if the rev. lr algorithm is correctly applied.

        Tests cases where both quote rule and tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by quote rule, two by tick rule, and two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [1, 1, 1, 0],
                [3, 2, 4, 4],
                [1, 1, np.nan, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        fitted_classifier = ClassicalClassifier(
            layers=[("rev_lr", subset)], random_state=42
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_emo(self, subset: str) -> None:
        """
        Test, if the emo algorithm is correctly applied.

        Tests cases where both quote rule at bid or ask and tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [
                    1,
                    1,
                    1,
                    0,
                ],
                [3, 2, 4, 4],
                [1, 1, np.inf, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        fitted_classifier = ClassicalClassifier(
            layers=[("emo", subset)], random_state=42
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_emo(self, subset: str) -> None:
        """
        Test, if the rev. emo algorithm is correctly applied.

        Tests cases where both quote rule at bid or ask and rev. tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [1, 1, 3, 0],
                [3, 1, 3, 0],
                [1, 1, 1, 0],
                [3, 2, 4, 4],
                [1, 1, np.inf, np.nan],
                [1, 1, np.nan, np.nan],
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_test = pd.Series([-1, 1, 1, -1, -1, 1])
        fitted_classifier = ClassicalClassifier(
            layers=[("rev_emo", subset)], random_state=42
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_clnv(self, subset: str) -> None:
        """
        Test, if the clnv algorithm is correctly applied.

        Tests cases where both quote rule and  tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by quote rule, two by tick rule, two by random chance.
        x_test = pd.DataFrame(
            [
                [5, 3, 1, 0],  # tick rule
                [0, 3, 1, 1],  # tick rule
                [2.9, 3, 1, 1],  # quote rule
                [2.3, 3, 1, 3],  # tick rule
                [1.7, 3, 1, 0],  # tick rule
                [1.3, 3, 1, 1],  # quote rule
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lag"],
        )
        y_test = pd.Series([1, -1, 1, -1, 1, -1])
        fitted_classifier = ClassicalClassifier(
            layers=[("clnv", subset)], random_state=42
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    @pytest.mark.parametrize("subset", ["best", "ex"])
    def test_rev_clnv(self, subset: str) -> None:
        """
        Test, if the rev. clnv algorithm is correctly applied.

        Tests cases where both quote rule and rev. tick rule all are used.

        Args:
            subset (str): subset e. g., 'ex'
        """
        x_train = pd.DataFrame(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_train = pd.Series([0, 0, 0])
        # .
        x_test = pd.DataFrame(
            [
                [5, 3, 1, 0],  # rev tick rule
                [0, 3, 1, 1],  # rev tick rule
                [2.9, 3, 1, 1],  # quote rule
                [2.3, 3, 1, 3],  # rev tick rule
                [1.7, 3, 1, 0],  # rev tick rule
                [1.3, 3, 1, 1],  # quote rule
            ],
            columns=["TRADE_PRICE", f"ask_{subset}", f"bid_{subset}", "price_all_lead"],
        )
        y_test = pd.Series([1, -1, 1, -1, 1, -1])
        fitted_classifier = ClassicalClassifier(
            layers=[("rev_clnv", subset)], random_state=5
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    def test_trade_size(self) -> None:
        """
        Test, if the trade size algorithm is correctly applied.

        Tests cases where relevant data is present or missing.
        """
        x_train = pd.DataFrame(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            columns=["TRADE_SIZE", "ask_size_ex", "bid_size_ex"],
        )
        y_train = pd.Series([0, 0, 0])
        # first two by trade size, random, at bid size, random, random.
        x_test = pd.DataFrame(
            [
                [1, 1, 3],
                [3, 1, 3],
                [1, 1, 1],
                [3, np.nan, 3],
                [1, np.inf, 2],
                [1, np.inf, 2],
            ],
            columns=["TRADE_SIZE", "ask_size_ex", "bid_size_ex"],
        )
        y_test = pd.Series([-1, 1, -1, 1, -1, 1])
        fitted_classifier = ClassicalClassifier(
            layers=[("trade_size", "ex")], random_state=42
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()

    def test_depth(self) -> None:
        """
        Test, if the depth rule is correctly applied.

        Tests cases where relevant data is present or missing.
        """
        x_train = pd.DataFrame(
            [[2, 1, 4, 4, 4], [1, 2, 2, 4, 3], [2, 1, 2, 4, 2], [1, 2, 2, 4, 2]],
            columns=[
                "ask_size_ex",
                "bid_size_ex",
                "ask_ex",
                "bid_ex",
                "TRADE_PRICE",
            ],
        )
        y_train = pd.Series([0, 0, 0, 0])
        # first three by depth, all other random as mid is different from trade price.
        x_test = pd.DataFrame(
            [
                [2, 1, 2, 4, 3],
                [1, 2, 2, 4, 3],
                [2, 1, 4, 4, 4],
                [2, 1, 2, 4, 2],
                [2, 1, 2, 4, 2],
            ],
            columns=[
                "ask_size_ex",
                "bid_size_ex",
                "ask_ex",
                "bid_ex",
                "TRADE_PRICE",
            ],
        )
        y_test = pd.Series([1, -1, 1, 1, -1])
        fitted_classifier = ClassicalClassifier(
            layers=[("depth", "ex")], random_state=5
        ).fit(x_train, y_train)
        y_pred = fitted_classifier.predict(x_test)
        assert (y_pred == y_test).all()
