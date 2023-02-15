import torch

from transformers import BertModel
from niftylittlepenguin.qa.constants import MODEL


class QABert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = BertModel.from_pretrained(MODEL)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, x):
        h = self.encoder(x)
        y = self.classifier(h)
        return y
