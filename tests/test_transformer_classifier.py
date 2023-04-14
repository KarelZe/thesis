"""
Tests for the transformer classifier.

Use of artificial data to test the classifier.
"""


from tests.templates import ClassifierMixin


class TestTransformerClassifier(ClassifierMixin):
    """
    Perform automated tests for TransformerClassifier.

    Args:
        unittest (_type_): unittest module
    """

    # def setup(self) -> None:
    #     """
    #     Set up basic classifier and data.

    #     Prepares inputs and expected outputs for testing.
    #     """
    #     self.x_train = pd.DataFrame(
    #         [[1, 2], [3, 4], [1, 2], [3, 4]], columns=["BEST_ASK", "BEST_BID"]
    #     )
    #     self.y_train = pd.Series([1, 1, -1, -1])
    #     self.x_test = pd.DataFrame(
    #         [[1, 2], [3, 4], [1, 2], [3, 4]], columns=["BEST_ASK", "BEST_BID"]
    #     )
    #     self.y_test = pd.Series([1, -1, 1, -1])

    #     dl_params = {
    #         "batch_size": 8,
    #         "shuffle": False,
    #         "device": "cpu",
    #     }

    #     module_params = {
    #         "depth": 1,
    #         "heads": 2,
    #         "dim": 2,
    #         "dim_out": 1,
    #         "mlp_act": nn.ReLU,
    #         "mlp_hidden_mults": (4, 2),
    #         "attn_dropout": 0.5,
    #         "ff_dropout": 0.5,
    #         "cat_features": [],
    #         "cat_cardinalities": (),
    #         "num_continuous": self.x_train.shape[1],
    #     }

    #     optim_params = {"lr": 0.1, "weight_decay": 1e-3}

    #     with patch.object(TransformerClassifier, "epochs", 5):
    #         self.clf = TransformerClassifier(
    #             module=FTTransformer,  # type: ignore
    #             module_params=module_params,
    #             optim_params=optim_params,
    #             dl_params=dl_params,
    #             callbacks=CallbackContainer([]),
    #         ).fit(self.x_train, self.y_train)

    # def test_sklearn_compatibility(self) -> None:
    #     """
    #     Test, if classifier is compatible with sklearn.
    #     """
    #     with patch.object(TransformerClassifier, "epochs", 1):
    #         check_estimator(self.clf)
