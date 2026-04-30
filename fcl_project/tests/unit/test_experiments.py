"""Unit tests for the experiments package."""

import unittest
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.experiments.metrics import AccuracyMatrix, evaluate_all_tasks
from code.experiments.federated import (
    FedAvgServer, FedProxServer, fedprox_local_loss, train_client_local,
)
from code.experiments.continual import (
    FineTuneStrategy, EWCStrategy, DERStrategy, FeatureDERBuffer,
)


class _SimpleModel(nn.Module):
    """Tiny model for testing."""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({
            'task_0': nn.Linear(8, 3),
            'task_1': nn.Linear(8, 2),
        })
        self.fc = nn.Linear(4, 8)

    def extract_features(self, x):
        return self.fc(x)

    def forward(self, x, task_id=0):
        feat = self.extract_features(x)
        return self.heads[f'task_{task_id}'](feat)


# ---------------------------------------------------------------------------
# AccuracyMatrix
# ---------------------------------------------------------------------------

class TestAccuracyMatrix(unittest.TestCase):
    def test_bwt_no_forgetting(self):
        am = AccuracyMatrix(3)
        for i in range(3):
            for j in range(3):
                am.update(i, j, 0.9)
        self.assertAlmostEqual(am.bwt(), 0.0, places=5)

    def test_bwt_with_forgetting(self):
        am = AccuracyMatrix(2)
        am.update(0, 0, 0.9)
        am.update(1, 0, 0.7)
        am.update(1, 1, 0.85)
        self.assertAlmostEqual(am.bwt(), -0.2, places=5)

    def test_fwt_positive(self):
        am = AccuracyMatrix(2)
        am.update(0, 0, 0.8)
        am.update(0, 1, 0.6)
        am.update(1, 0, 0.75)
        am.update(1, 1, 0.85)
        fwt = am.fwt(baselines=np.array([0.0, 0.0]))
        self.assertAlmostEqual(fwt, 0.6, places=5)

    def test_average_accuracy(self):
        am = AccuracyMatrix(2)
        am.update(0, 0, 0.8)
        am.update(1, 0, 0.7)
        am.update(1, 1, 0.9)
        self.assertAlmostEqual(am.average_accuracy(), 0.8, places=5)

    def test_forgetting(self):
        am = AccuracyMatrix(2)
        am.update(0, 0, 0.9)
        am.update(1, 0, 0.7)
        am.update(1, 1, 0.85)
        self.assertAlmostEqual(am.forgetting(), 0.2, places=5)

    def test_to_dict(self):
        am = AccuracyMatrix(2)
        am.update(0, 0, 0.8)
        am.update(1, 1, 0.9)
        d = am.to_dict()
        self.assertIn('bwt', d)
        self.assertIn('matrix', d)


# ---------------------------------------------------------------------------
# FedAvg / FedProx
# ---------------------------------------------------------------------------

class TestFedAvgServer(unittest.TestCase):
    def test_aggregate_uniform(self):
        model = nn.Linear(4, 2)
        server = FedAvgServer(model, device='cpu')

        sd1 = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
        sd2 = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        server.aggregate([sd1, sd2])

        for v in server.global_model.state_dict().values():
            self.assertTrue(torch.allclose(v, torch.ones_like(v) * 0.5, atol=1e-5))

    def test_aggregate_weighted(self):
        model = nn.Linear(4, 2)
        server = FedAvgServer(model, device='cpu')

        sd1 = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
        sd2 = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        server.aggregate([sd1, sd2], client_weights=[3.0, 1.0])

        for v in server.global_model.state_dict().values():
            self.assertTrue(torch.allclose(v, torch.ones_like(v) * 0.75, atol=1e-5))

    def test_distribute(self):
        model = nn.Linear(4, 2)
        server = FedAvgServer(model, device='cpu')
        clients = [nn.Linear(4, 2) for _ in range(3)]
        server.distribute(clients)
        for cm in clients:
            for k in model.state_dict():
                self.assertTrue(torch.equal(cm.state_dict()[k], model.state_dict()[k]))


class TestFedProxLoss(unittest.TestCase):
    def test_zero_when_equal(self):
        model = nn.Linear(4, 2)
        gp = {k: v.clone() for k, v in model.state_dict().items()}
        loss = fedprox_local_loss(model, gp, mu=0.01)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_when_diverged(self):
        model = nn.Linear(4, 2)
        gp = {k: v.clone() + 1.0 for k, v in model.state_dict().items()}
        loss = fedprox_local_loss(model, gp, mu=0.01)
        self.assertGreater(loss.item(), 0.0)


# ---------------------------------------------------------------------------
# Continual strategies
# ---------------------------------------------------------------------------

class TestFineTuneStrategy(unittest.TestCase):
    def test_no_regularization(self):
        ft = FineTuneStrategy()
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 3, (4,))
        loss = ft.compute_loss(None, logits, labels, 0)
        self.assertEqual(loss.item(), 0.0)


class TestEWCStrategy(unittest.TestCase):
    def test_fisher_computation(self):
        model = _SimpleModel()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,))),
            batch_size=8,
        )
        ewc = EWCStrategy(lambda_ewc=0.5, device='cpu')
        ewc.post_task(model, 0, loader)
        self.assertTrue(len(ewc.fisher) > 0)
        self.assertTrue(len(ewc.optimal_params) > 0)

    def test_ewc_loss_after_fisher(self):
        model = _SimpleModel()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, 4), torch.randint(0, 3, (16,))),
            batch_size=8,
        )
        ewc = EWCStrategy(lambda_ewc=0.5, device='cpu')
        ewc.post_task(model, 0, loader)

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        logits = torch.randn(4, 3)
        loss = ewc.compute_loss(model, logits, torch.randint(0, 3, (4,)), 1)
        self.assertGreater(loss.item(), 0.0)


class TestFeatureDERBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        buf = FeatureDERBuffer(buffer_size=100, device='cpu')
        buf.add(
            torch.randn(16, 576),
            torch.randn(16, 9),
            torch.randint(0, 9, (16,)),
            torch.zeros(16, dtype=torch.long),
        )
        batch = buf.sample(8)
        self.assertIsNotNone(batch)
        self.assertEqual(batch['features'].shape, (8, 576))

    def test_reservoir_overflow(self):
        buf = FeatureDERBuffer(buffer_size=10, device='cpu')
        for _ in range(20):
            buf.add(
                torch.randn(5, 32),
                torch.randn(5, 4),
                torch.randint(0, 4, (5,)),
                torch.zeros(5, dtype=torch.long),
            )
        self.assertEqual(buf.n_samples, 100)
        batch = buf.sample(10)
        self.assertIsNotNone(batch)
        self.assertEqual(batch['features'].shape[0], 10)


# ---------------------------------------------------------------------------
# MultiHeadImageModel (lightweight test without pretrained weights)
# ---------------------------------------------------------------------------

class TestMultiHeadModel(unittest.TestCase):
    def test_forward_per_task(self):
        model = _SimpleModel()
        x = torch.randn(4, 4)
        out0 = model(x, task_id=0)
        out1 = model(x, task_id=1)
        self.assertEqual(out0.shape, (4, 3))
        self.assertEqual(out1.shape, (4, 2))

    def test_extract_features(self):
        model = _SimpleModel()
        x = torch.randn(4, 4)
        feat = model.extract_features(x)
        self.assertEqual(feat.shape, (4, 8))


if __name__ == '__main__':
    unittest.main()
