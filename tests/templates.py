"""Tests for Neural networks.

See:
https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
http://karpathy.github.io/2019/04/25/recipe/
https://krokotsch.eu/posts/deep-learning-unit-tests/
"""

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator


class NeuralNetTestsMixin:
    """Perform automated tests for neural networks.

    Args:
    ----
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    # https://mypy.readthedocs.io/en/stable/protocols.html
    # https://stackoverflow.com/a/67679462/5755604
    net: nn.Module
    x_cat: torch.Tensor
    x_cont: torch.Tensor
    expected_outputs: torch.Tensor
    batch_size: int

    def get_outputs(self) -> torch.Tensor:
        """Return relevant output of model.

        Returns:
            torch.Tensor: outputs
        """
        return self.net(self.x_cat.clone(), self.x_cont.clone())

    @torch.no_grad()
    def test_shapes(self) -> None:
        """Test, if shapes of the network equal the targets.

        Loss might be calculated due to broadcasting, but might be wrong.
        Adapted from: # https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        outputs = self.get_outputs()
        assert self.expected_outputs.shape == outputs.shape

    def test_convergence(self) -> None:
        """Tests, whether loss approaches zero for single batch.

        Training on a single batch leads to serious overfitting.
        If loss does not become, this indicates a possible error.
        See: http://karpathy.github.io/2019/04/25/recipe/
        """
        optimizer = optim.Adam(self.net.parameters(), lr=3e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.net.train()

        # perform training
        for _ in range(512):
            outputs = self.get_outputs()
            optimizer.zero_grad()

            loss = criterion(outputs, self.expected_outputs)

            loss.backward()
            optimizer.step()

        print(loss.detach().cpu().numpy())
        assert loss.detach().cpu().numpy() <= 5e-3

    @torch.no_grad()
    @pytest.mark.skipif(
        torch.cuda.is_available() is False, reason="No GPU was detected."
    )
    def test_device_moving(self) -> None:
        """Test, if all tensors reside on the gpu / cpu.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        net_on_gpu = self.net.to("cuda:0")
        net_back_on_cpu = net_on_gpu.cpu()

        outputs_cpu = self.net(self.x_cat, self.x_cont)
        outputs_gpu = net_on_gpu(self.x_cat.to("cuda:0"), self.x_cont.to("cuda:0"))
        outputs_back_on_cpu = net_back_on_cpu(self.x_cat, self.x_cont)

        assert round(abs(0.0 - torch.sum(outputs_cpu - outputs_gpu.cpu())), 7) == 0
        assert round(abs(0.0 - torch.sum(outputs_cpu - outputs_back_on_cpu)), 7) == 0

    def test_all_parameters_updated(self) -> None:
        """Test, if all parameters are updated.

        If parameters are not updated this could indicate dead ends.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=3e-4)

        outputs = self.get_outputs()
        loss = outputs.mean()
        loss.backward()
        optimizer.step()

        for _, param in self.net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                param_sum = torch.sum(param.grad**2)
                assert torch.tensor(0) != param_sum

    def test_batch_independence(self) -> None:
        """Checks sample independence by performing of inputs.

        Required as SGD-based algorithms like ADAM work on mini-batches. Batching
        training samples assumes that your model can process each sample as if they
        were fed individually. In other words, the samples in  your batch do not
        influence each other when processed. This assumption is a brittle one and can
        break with one misplaced reshape or aggregation over a wrong tensor dimension.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        # no gradient for int tensors, only for float tensors
        self.x_cat.requires_grad = False
        self.x_cont.requires_grad = True
        # Compute forward pass in eval mode to deactivate batch norm

        self.net.eval()
        outputs = self.get_outputs()
        self.net.train()

        # Mask loss for certain samples in batch
        mask_idx = torch.randint(0, self.batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check if gradient exists and is zero for masked samples.
        # Test only for float tensors, as int tensors do not have gradients.
        for i, grad in enumerate(self.x_cont.grad):
            if i == mask_idx:
                assert torch.all(grad == 0).item()
            else:
                assert not torch.all(grad == 0)


class ClassifierMixin:
    """Perform automated tests for Classifiers.

    Args:
    ----
        unittest (_type_): unittest module
    """

    clf: BaseEstimator
    x_test: pd.DataFrame
    y_test: pd.Series

    def test_sklearn_compatibility(self) -> None:
        """Test, if classifier is compatible with sklearn."""
        check_estimator(self.clf)

    def test_shapes(self) -> None:
        """Test, if shapes of the classifier equal the targets.

        Shapes are usually [no. of samples, 1].
        """
        y_pred = self.clf.predict(self.x_test)

        assert self.y_test.shape == y_pred.shape

    def test_proba(self) -> None:
        """Test, if probabilities are in [0, 1]."""
        y_pred = self.clf.predict_proba(self.x_test)
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()

    def test_score(self) -> None:
        """Test, if score is correctly calculated..

        For a random classification i. e., `layers=[("nan", "ex")]`, the score
        should be around 0.5.
        """
        accuracy = self.clf.score(self.x_test, self.y_test)
        assert 0.0 <= accuracy <= 1.0
