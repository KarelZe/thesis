"""
Tests for Neural networks.

See:
https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
http://karpathy.github.io/2019/04/25/recipe/
https://krokotsch.eu/posts/deep-learning-unit-tests/
"""
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from otc.models.objective import set_seed
from otc.models.tabtransformer import TabTransformer


class TestNN(unittest.TestCase):
    """
    Perform automated tests for neural networks.

    Args:
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def get_outputs(self) -> torch.Tensor:
        """
        Return relevant output of model.

        Returns:
            torch.Tensor: outputs
        """
        # TODO: Find out why net changes input.
        self.x_cat = torch.randint(0, 1, (self.batch_size, self.num_features_cat)).to(
            "cpu"
        )

        outputs = self.net(self.x_cat, self.x_cont)  # type: ignore
        return outputs

    def setUp(self) -> None:
        """
        Set up basic network and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features_cont = 5
        self.num_features_cat = 1
        self.num_unique_cat = tuple([2])
        self.batch_size = 64
        self.epochs = 256
        self.threshold = 1e-3

        set_seed()

        # lstm moves to device autoamtically, if available. see lstm.py
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_cat = torch.randint(0, 1, (self.batch_size, self.num_features_cat)).to(
            device
        )
        self.x_cont = (
            torch.randn(self.batch_size, self.num_features_cont).float().to(device)
        )
        self.expected_outputs = (
            torch.randint(0, 1, (self.batch_size, 1)).float().to(device)
        )

        self.net = TabTransformer(
            categories=self.num_unique_cat,
            num_continuous=self.num_features_cont,
            dim_out=1,
            mlp_act=nn.ReLU(),
            dim=32,
            depth=2,
            heads=6,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
        ).to(device)

    @torch.no_grad()
    def test_shapes(self) -> None:
        """
        Test, if shapes of the network equal the targets.

        Loss might be calculated due to broadcasting, but might be wrong.
        Adapted from: # https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        outputs = self.get_outputs()
        self.assertEqual(self.expected_outputs.shape, outputs.shape)

    def test_convergence(self) -> None:
        """
        Tests, whether loss approaches zero for single batch.

        Training on a single batch leads to serious overfitting.
        If loss does not become, this indicates a possible error.
        See: http://karpathy.github.io/2019/04/25/recipe/
        """
        optimizer = optim.Adam(self.net.parameters(), lr=3e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.net.train()

        # perform training
        for _ in range(self.epochs):

            outputs = self.get_outputs()
            optimizer.zero_grad()

            loss = criterion(outputs, self.expected_outputs)

            loss.backward()
            optimizer.step()

        self.assertLessEqual(loss.detach().cpu().numpy(), self.threshold)

    @unittest.skipIf(not torch.cuda.is_available(), reason="Skip. No gpu found.")
    def test_device_moving(self) -> None:
        """
        Test, if all tensors reside on the gpu / cpu.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        net_on_gpu = self.net.to("cuda:0")
        net_back_on_cpu = net_on_gpu.cpu()

        outputs_cpu = self.net(self.x_cat, self.x_cont)
        outputs_gpu = net_on_gpu(self.x_cat.to("cuda:0"), self.x_cont.to("cuda:0"))
        outputs_back_on_cpu = net_back_on_cpu(self.x_cat, self.x_cont)

        self.assertAlmostEqual(0.0, torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0.0, torch.sum(outputs_cpu - outputs_back_on_cpu))

    def test_all_parameters_updated(self) -> None:
        """
        Test, if all parameters are updated.

        If parameters are not updated this could indicate dead ends.
        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=3e-4)

        outputs = self.get_outputs()
        loss = outputs.mean()
        loss.backward()
        optimizer.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    param_sum = torch.sum(param.grad**2)
                    self.assertNotEqual(torch.tensor(0), param_sum)
