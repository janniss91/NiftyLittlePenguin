import pytest
import torch

from niftylittlepenguin.qa.metrics import QAMetrics


class TestQAMetrics:
    @pytest.fixture
    def metrics_obj(self):
        # (batch=3, seq_len=3)
        start_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
        end_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        start_gold = torch.tensor([2, 1, 1])
        end_gold = torch.tensor([2, 0, 1])

        return QAMetrics(
            start_logits=start_logits,
            end_logits=end_logits,
            start_gold=start_gold,
            end_gold=end_gold,
        )

    @pytest.fixture
    def metrics_obj_all_wrong(self):
        # (batch=3, seq_len=3)
        start_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
        end_logits = torch.tensor([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        start_gold = torch.tensor([2, 1, 0])
        end_gold = torch.tensor([1, 0, 1])

        return QAMetrics(
            start_logits=start_logits,
            end_logits=end_logits,
            start_gold=start_gold,
            end_gold=end_gold,
        )

    def test_acc_whole_question(self, metrics_obj):
        assert metrics_obj.acc_whole_question() == 2 / 3

    def test_acc_whole_question_all_wrong(self, metrics_obj_all_wrong):
        assert metrics_obj_all_wrong.acc_whole_question() == 0
