from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from transformers import BertModel

from niftylittlepenguin.qa.config import LOG_INTERVAL
from niftylittlepenguin.qa.constants import MODEL
from niftylittlepenguin.qa.metrics import QAMetrics


class LitQABert(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = BertModel.from_pretrained(MODEL)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.qa_metrics = QAMetrics()
        self.n_train_steps = 0
        self.n_dev_steps = 0

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**inputs).last_hidden_state
        logits = self.classifier(hidden)

        return logits

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, accuracies = self._step(batch, "train")
        self.n_train_steps += 1

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(accuracies, on_step=True)
        self.print_metrics(accuracies, "train")

        return loss
    
    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:
        # TODO: Store model checkpoints.
        loss, accuracies = self._step(batch, "dev")

        self.n_dev_steps += 1
        self.log(
            "dev_loss", loss, on_epoch=True, prog_bar=True, logger=False
        )
        self.log_dict(accuracies, on_step=True)
        self.print_metrics(accuracies, "dev")
        
        return loss

    def _step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], split) -> torch.Tensor:
        inputs, labels = batch
        logits = self(inputs)

        # Separate the predictions for start and end.
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]

        # Separate the true starts and ends.
        starts = labels[:, 0]
        ends = labels[:, 1]

        # Calculate the losses and add them together.
        start_loss = self.loss_fn(start_logits, starts)
        end_loss = self.loss_fn(end_logits, ends)
        loss = start_loss + end_loss

        accuracies = self.qa_metrics.accuracies(start_logits.cpu(), end_logits.cpu(), starts.cpu(), ends.cpu(), split)

        return loss, accuracies
    
    def on_validation_end(self):
        print("Total validation accuracy:\n")
        print(self.qa_metrics.final_accuracies())
        
    
    def print_metrics(self, accuracies: Dict[str, torch.Tensor], split: str):
        if split == "train" and self.n_train_steps and self.n_train_steps % LOG_INTERVAL == 0:
            print(f"\nTrain metrics: {accuracies}")
        elif split == "dev" and self.n_dev_steps and self.n_dev_steps % LOG_INTERVAL == 0:
            print(f"\nDev metrics: {accuracies}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


class QABert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = BertModel.from_pretrained(MODEL)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, x):
        h = self.encoder(**x).last_hidden_state
        y = self.classifier(h)
        return y
