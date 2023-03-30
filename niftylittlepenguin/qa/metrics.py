from typing import Dict

import torch
from torchmetrics import Accuracy


class QAMetrics:
    def __init__(self) -> None:

        self.all_pred_starts = torch.tensor([])
        self.all_gold_starts = torch.tensor([])
        self.all_pred_ends = torch.tensor([])
        self.all_gold_ends = torch.tensor([])

    # TODO: Show single dev outputs to see what samples are classified incorrectl.

    def accuracies(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_gold: torch.Tensor,
        end_gold: torch.Tensor,
        split: str = "train",
    ) -> Dict[str, int]:

        # Shape: (batch_size)
        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)

        if split == "dev":
            self.compile_outputs(start_pred, end_pred, start_gold, end_gold)

        output = {
            "acc_starts": self.accuracy(start_pred, start_gold).item(),
            "acc_ends": self.accuracy(end_pred, end_gold).item(),
            "acc_whole_question": self.acc_whole_question(
                starts_pred=start_pred,
                ends_pred=end_pred,
                starts_gold=start_gold,
                ends_gold=end_gold,
            ).item(),
        }

        return output
    
    def accuracy(self, pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
        return (pred == gold).sum() / pred.shape[0]

    def acc_whole_question(
        self,
        starts_pred: torch.Tensor,
        ends_pred: torch.Tensor,
        starts_gold: torch.Tensor,
        ends_gold: torch.Tensor,
    ) -> torch.Tensor:
        both_correct = (starts_pred == starts_gold) * (ends_pred == ends_gold)
        return both_correct.sum() / both_correct.shape[0]

    def compile_outputs(self, start_pred: torch.Tensor, end_pred: torch.Tensor, start_gold: torch.Tensor, end_gold: torch.Tensor):
        self.all_pred_starts = torch.cat((self.all_pred_starts, start_pred))
        self.all_gold_starts = torch.cat((self.all_gold_starts, start_gold))
        self.all_pred_ends = torch.cat((self.all_pred_ends, end_pred))
        self.all_gold_ends = torch.cat((self.all_gold_ends, end_gold))

    def final_accuracies(self) -> Dict[str, int]:
        start_acc = self.accuracy(self.all_pred_starts, self.all_gold_starts)
        end_acc = self.accuracy(self.all_pred_ends, self.all_gold_ends)
        whole_acc = self.acc_whole_question(self.all_pred_starts, self.all_pred_ends, self.all_gold_starts, self.all_gold_ends)

        self.reset()
        return {
            "acc_starts": start_acc.item(),
            "acc_ends": end_acc.item(),
            "acc_whole_question": whole_acc.item(),
        }
    
    def reset(self):
        self.all_pred_starts = torch.tensor([])
        self.all_gold_starts = torch.tensor([])
        self.all_pred_ends = torch.tensor([])
        self.all_gold_ends = torch.tensor([])
