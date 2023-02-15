import json
from typing import List
from torch.utils.data import Dataset

from niftylittlepenguin.qa.encode_data import SQuADEncoding


class QADataset(Dataset):
    def __init__(self, qa_data: list, split: str):
        self.split = split
        self.qa_data = qa_data

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, index):
        return self.qa_data[index]


class SQuAD_Dataset(QADataset):
    def __init__(self, qa_data: List[SQuADEncoding], split: str):
        super().__init__(qa_data, split)

    def __getitem__(self, index):
        instance = self.qa_data[index]

        x = {
            "input_ids": instance.input_ids,
            "token_type_ids": instance.token_type_ids,
            "attention_mask": instance.attention_mask,
        }
        y = instance.answer_offsets

        return x, y
