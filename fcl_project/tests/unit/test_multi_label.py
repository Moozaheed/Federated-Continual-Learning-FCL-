"""Unit tests for multi-label classification support.

Tests that the experiment pipeline correctly handles multi-label tasks
(ChestMNIST, MIMIC-CXR) alongside single-label tasks.
"""

import unittest
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/bs01233/Documents/FL/fcl_project')

from code.experiments.run_experiments import (
    MultiHeadImageModel, get_criterion, load_task_datasets,
)
from code.experiments.metrics import (
    evaluate_all_tasks, compute_roc_auc_per_task,
    _extract_batch, _evaluate_multi_label, _evaluate_single_label,
)
from code.experiments.federated import (
    train_client_local, create_dirichlet_splits,
)
from code.experiments.continual import (
    FineTuneStrategy, EWCStrategy, DERStrategy, GenReplayStrategy,
    FeatureDERBuffer, _extract_cl_batch,
)


class TestGetCriterion(unittest.TestCase):
    """Test loss function selection."""

    def test_single_label_returns_ce(self):
        criterion = get_criterion('single_label')
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)

    def test_multi_label_returns_bce(self):
        criterion = get_criterion('multi_label')
        self.assertIsInstance(criterion, nn.BCEWithLogitsLoss)


class TestMultiHeadModel(unittest.TestCase):
    """Test MultiHeadImageModel with mixed task types."""

    def setUp(self):
        self.task_classes = [9, 8, 14]
        self.task_types = ['single_label', 'single_label', 'multi_label']
        self.model = MultiHeadImageModel(
            self.task_classes, pretrained=False, task_types=self.task_types
        )

    def test_model_has_task_types(self):
        self.assertEqual(self.model.task_types, self.task_types)

    def test_single_label_head_output(self):
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x, task_id=0)
        self.assertEqual(out.shape, (4, 9))

    def test_multi_label_head_output(self):
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = self.model(x, task_id=2)
        self.assertEqual(out.shape, (4, 14))

    def test_bce_loss_with_multi_label_head(self):
        x = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4, 14)).float()
        logits = self.model(x, task_id=2)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels)
        self.assertTrue(loss.item() > 0)
        loss.backward()

    def test_ce_loss_with_single_label_head(self):
        x = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 9, (4,))
        logits = self.model(x, task_id=0)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        self.assertTrue(loss.item() > 0)
        loss.backward()


class TestExtractBatch(unittest.TestCase):
    """Test batch extraction helper."""

    def test_extract_single_label_dict(self):
        batch = {
            'image': torch.randn(4, 3, 224, 224),
            'label': torch.randint(0, 9, (4,)),
        }
        images, labels = _extract_batch(batch, 'cpu', 'single_label')
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(labels.shape, (4,))

    def test_extract_multi_label_dict(self):
        batch = {
            'image': torch.randn(4, 3, 224, 224),
            'labels': torch.randint(0, 2, (4, 14)).float(),
        }
        images, labels = _extract_batch(batch, 'cpu', 'multi_label')
        self.assertEqual(images.shape, (4, 3, 224, 224))
        self.assertEqual(labels.shape, (4, 14))
        self.assertEqual(labels.dtype, torch.float32)

    def test_extract_tuple_batch(self):
        batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 9, (4,)))
        images, labels = _extract_batch(batch, 'cpu', 'single_label')
        self.assertEqual(images.shape, (4, 3, 224, 224))

    def test_cl_extract_multi_label(self):
        batch = {
            'image': torch.randn(4, 3, 224, 224),
            'labels': torch.randint(0, 2, (4, 14)).float(),
        }
        images, labels = _extract_cl_batch(batch, 'cpu', 'multi_label')
        self.assertEqual(labels.shape, (4, 14))


class TestEvaluateMultiLabel(unittest.TestCase):
    """Test multi-label evaluation metrics."""

    def setUp(self):
        self.model = MultiHeadImageModel([9, 14], pretrained=False,
                                          task_types=['single_label', 'multi_label'])
        self.model.eval()

    def test_evaluate_single_label(self):
        dataset = _FakeDataset(100, 9, multi_label=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        score = _evaluate_single_label(self.model, loader, 0, 'cpu')
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_multi_label(self):
        dataset = _FakeDataset(100, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        score = _evaluate_multi_label(self.model, loader, 1, 'cpu')
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_all_mixed(self):
        ds_sl = _FakeDataset(50, 9, multi_label=False)
        ds_ml = _FakeDataset(50, 14, multi_label=True)
        loaders = [
            torch.utils.data.DataLoader(ds_sl, batch_size=32),
            torch.utils.data.DataLoader(ds_ml, batch_size=32),
        ]
        scores = evaluate_all_tasks(
            self.model, loaders, 'cpu',
            task_types=['single_label', 'multi_label'],
        )
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_roc_auc_mixed(self):
        ds_sl = _FakeDataset(50, 9, multi_label=False)
        ds_ml = _FakeDataset(50, 14, multi_label=True)
        loaders = [
            torch.utils.data.DataLoader(ds_sl, batch_size=32),
            torch.utils.data.DataLoader(ds_ml, batch_size=32),
        ]
        scores = compute_roc_auc_per_task(
            self.model, loaders, 'cpu',
            task_types=['single_label', 'multi_label'],
        )
        self.assertEqual(len(scores), 2)


class TestDERBufferMultiLabel(unittest.TestCase):
    """Test DER buffer with multi-label data."""

    def test_buffer_stores_multi_label_as_argmax(self):
        buf = FeatureDERBuffer(buffer_size=100)
        features = torch.randn(10, 576)
        logits = torch.randn(10, 14)
        labels = torch.randint(0, 2, (10, 14)).float()
        task_ids = torch.zeros(10, dtype=torch.long)
        buf.add(features, logits, labels, task_ids)
        sample = buf.sample(5)
        self.assertIsNotNone(sample)
        self.assertEqual(sample['labels'].dim(), 1)
        self.assertTrue((sample['labels'] >= 0).all())
        self.assertTrue((sample['labels'] < 14).all())

    def test_buffer_stores_single_label(self):
        buf = FeatureDERBuffer(buffer_size=100)
        features = torch.randn(10, 576)
        logits = torch.randn(10, 9)
        labels = torch.randint(0, 9, (10,))
        task_ids = torch.zeros(10, dtype=torch.long)
        buf.add(features, logits, labels, task_ids)
        sample = buf.sample(5)
        self.assertEqual(sample['labels'].dim(), 1)

    def test_buffer_mixed_tasks(self):
        buf = FeatureDERBuffer(buffer_size=200)
        feat_sl = torch.randn(20, 576)
        logits_sl = torch.randn(20, 9)
        labels_sl = torch.randint(0, 9, (20,))
        tid_sl = torch.zeros(20, dtype=torch.long)
        buf.add(feat_sl, logits_sl, labels_sl, tid_sl)

        feat_ml = torch.randn(20, 576)
        logits_ml = torch.randn(20, 14)
        labels_ml = torch.randint(0, 2, (20, 14)).float()
        tid_ml = torch.ones(20, dtype=torch.long)
        buf.add(feat_ml, logits_ml, labels_ml, tid_ml)

        sample = buf.sample(10)
        self.assertIsNotNone(sample)
        self.assertEqual(sample['features'].shape[1], 576)


class TestTrainClientMultiLabel(unittest.TestCase):
    """Test federated training with multi-label data."""

    def test_train_client_multi_label(self):
        model = MultiHeadImageModel([14], pretrained=False,
                                     task_types=['multi_label'])
        dataset = _FakeDataset(32, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        sd = train_client_local(
            model, loader, opt, criterion, 'cpu', 1,
            task_id=0, task_type='multi_label',
        )
        self.assertIsNotNone(sd)

    def test_train_client_single_label(self):
        model = MultiHeadImageModel([9], pretrained=False)
        dataset = _FakeDataset(32, 9, multi_label=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        sd = train_client_local(
            model, loader, opt, criterion, 'cpu', 1,
            task_id=0, task_type='single_label',
        )
        self.assertIsNotNone(sd)


class TestDirichletSplitsMultiLabel(unittest.TestCase):
    """Test Dirichlet splits with multi-label data."""

    def test_splits_multi_label(self):
        dataset = _FakeDataset(100, 14, multi_label=True)
        splits = create_dirichlet_splits(
            dataset, n_clients=4, alpha=0.5, task_type='multi_label',
        )
        self.assertEqual(len(splits), 4)
        total = sum(len(s) for s in splits)
        self.assertEqual(total, 100)

    def test_splits_single_label(self):
        dataset = _FakeDataset(100, 9, multi_label=False)
        splits = create_dirichlet_splits(
            dataset, n_clients=4, alpha=0.5, task_type='single_label',
        )
        self.assertEqual(len(splits), 4)
        total = sum(len(s) for s in splits)
        self.assertEqual(total, 100)


class TestEWCMultiLabel(unittest.TestCase):
    """Test EWC Fisher computation with multi-label data."""

    def test_ewc_post_task_multi_label(self):
        model = MultiHeadImageModel([14], pretrained=False,
                                     task_types=['multi_label'])
        dataset = _FakeDataset(32, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        ewc = EWCStrategy(lambda_ewc=0.5, device='cpu')
        ewc.post_task(model, 0, loader, task_type='multi_label')
        self.assertTrue(len(ewc.fisher) > 0)
        self.assertTrue(len(ewc.optimal_params) > 0)

    def test_ewc_post_task_single_label(self):
        model = MultiHeadImageModel([9], pretrained=False)
        dataset = _FakeDataset(32, 9, multi_label=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        ewc = EWCStrategy(lambda_ewc=0.5, device='cpu')
        ewc.post_task(model, 0, loader, task_type='single_label')
        self.assertTrue(len(ewc.fisher) > 0)

    def test_ewc_compute_loss_after_multi_label(self):
        model = MultiHeadImageModel([14, 9], pretrained=False,
                                     task_types=['multi_label', 'single_label'])
        ds_ml = _FakeDataset(32, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(ds_ml, batch_size=16)
        ewc = EWCStrategy(lambda_ewc=0.5, device='cpu')
        ewc.post_task(model, 0, loader, task_type='multi_label')
        # Perturb model so EWC loss is non-zero
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        x = torch.randn(4, 3, 224, 224)
        logits = model(x, task_id=1)
        labels = torch.randint(0, 9, (4,))
        loss = ewc.compute_loss(model, logits, labels, 1)
        self.assertTrue(loss.item() >= 0)


class TestDERMultiLabel(unittest.TestCase):
    """Test DER++ with multi-label data."""

    def test_der_post_task_multi_label(self):
        model = MultiHeadImageModel([14], pretrained=False,
                                     task_types=['multi_label'])
        dataset = _FakeDataset(32, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        der = DERStrategy(buffer_size=100, device='cpu')
        der.post_task(model, 0, loader, task_type='multi_label')
        self.assertTrue(der.buffer.n_samples > 0)

    def test_der_post_task_single_label(self):
        model = MultiHeadImageModel([9], pretrained=False)
        dataset = _FakeDataset(32, 9, multi_label=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        der = DERStrategy(buffer_size=100, device='cpu')
        der.post_task(model, 0, loader, task_type='single_label')
        self.assertTrue(der.buffer.n_samples > 0)


class TestGenReplayMultiLabel(unittest.TestCase):
    """Test Generative Replay with multi-label data."""

    def test_gen_replay_post_task_multi_label(self):
        model = MultiHeadImageModel([14], pretrained=False,
                                     task_types=['multi_label'])
        dataset = _FakeDataset(32, 14, multi_label=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        gr = GenReplayStrategy(feature_dim=576, device='cpu', vae_epochs=2)
        gr.post_task(model, 0, loader, task_type='multi_label')
        self.assertEqual(len(gr.generators), 1)
        self.assertEqual(gr.task_n_classes[0], 14)


class TestChestMNISTFormat(unittest.TestCase):
    """Test that ChestMNIST is properly flagged as multi-label."""

    def test_chest_info_is_multi_label(self):
        from code.datasets.medmnist import MedMNISTDataset
        info = MedMNISTDataset.get_info('chest')
        self.assertTrue(info.get('multi_label', False))
        self.assertEqual(info['n_classes'], 14)

    def test_other_datasets_single_label(self):
        from code.datasets.medmnist import MedMNISTDataset
        for ds in ['path', 'blood', 'derma', 'retina', 'tissue', 'organ']:
            info = MedMNISTDataset.get_info(ds)
            self.assertFalse(info.get('multi_label', False),
                             f"{ds} should be single_label")


# ---------------------------------------------------------------------------
# Fake dataset helper
# ---------------------------------------------------------------------------

class _FakeDataset(torch.utils.data.Dataset):
    """Minimal dataset for testing with configurable label format."""

    def __init__(self, n_samples: int, n_classes: int, multi_label: bool = False):
        self.n = n_samples
        self.n_classes = n_classes
        self.multi_label = multi_label
        self.images = torch.randn(n_samples, 3, 224, 224)
        if multi_label:
            self._labels = torch.randint(0, 2, (n_samples, n_classes)).float()
        else:
            self._labels = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.multi_label:
            return {'image': self.images[idx], 'labels': self._labels[idx]}
        return {'image': self.images[idx], 'label': self._labels[idx]}


if __name__ == '__main__':
    unittest.main()
