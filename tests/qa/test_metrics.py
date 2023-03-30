import pytest
import torch

from niftylittlepenguin.qa.metrics import QAMetrics


class TestQAMetrics:
    @pytest.fixture
    def metrics(self):
        # (batch=3, seq_len=3)
        start_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
        end_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        start_gold = torch.tensor([2, 1, 1])
        end_gold = torch.tensor([2, 0, 1])

        return start_logits, end_logits, start_gold, end_gold
    
    @pytest.fixture
    def metrics_all_wrong(self):
        start_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
        end_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        start_gold = torch.tensor([2, 1, 0])
        end_gold = torch.tensor([1, 0, 1])

        return start_logits, end_logits, start_gold, end_gold

    def test_acc_whole_question(self, metrics):
        start_logits, end_logits, start_gold, end_gold = metrics

        metrics_obj = QAMetrics()
        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)
        acc = metrics_obj.acc_whole_question(start_pred, end_pred, start_gold, end_gold)
        assert acc == 2 / 3

    def test_acc_whole_question_all_wrong(self, metrics_all_wrong):
        start_logits, end_logits, start_gold, end_gold = metrics_all_wrong

        metrics_obj = QAMetrics()
        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)
        acc = metrics_obj.acc_whole_question(start_pred, end_pred, start_gold, end_gold)

        assert acc == 0
