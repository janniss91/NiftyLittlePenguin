from typing import Dict
import torch
from torchmetrics import Accuracy


class QAMetrics:
    def __init__(self) -> None:
        self.latest_accs = None

    # TODO: Show single dev outputs to see what samples are classified incorrectl.

    def accuracies(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_gold: torch.Tensor,
        end_gold: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        acc = Accuracy(task="multiclass", num_classes=start_logits.shape[1])

        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)

        output = {
            "acc_starts": acc(start_pred, start_gold),
            "acc_ends": acc(end_pred, end_gold),
            "acc_whole_question": self.acc_whole_question(
                starts_pred=start_pred,
                ends_pred=end_pred,
                starts_gold=start_gold,
                ends_gold=end_gold,
            ),
        }
        
        self.latest_accs = output
        return output

    def acc_whole_question(
        self,
        starts_pred: torch.Tensor,
        ends_pred: torch.Tensor,
        starts_gold: torch.Tensor,
        ends_gold: torch.Tensor,
    ):
        both_correct = (starts_pred == starts_gold) * (ends_pred == ends_gold)
        return both_correct.sum() / both_correct.shape[0]

