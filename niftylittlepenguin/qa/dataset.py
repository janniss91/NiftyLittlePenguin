from abc import abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizerFast

from niftylittlepenguin.qa.encode_data import QADataEncoder, SQuADEncoder, SQuADEncoding
from niftylittlepenguin.qa.constants import MAX_LENGTH


class QADataset(IterableDataset):
    """
    The dataset is implemented as iterable dataset to work on machines with small memory.
    """
    def __init__(self, data_encoder: QADataEncoder, tokenizer: PreTrainedTokenizerFast, split: str):
        self.data_encoder = data_encoder
        self.tokenizer = tokenizer
        self.split = split

    @abstractmethod
    def __iter__(self):
        pass


class SQuAD_Dataset(QADataset):
    def __init__(
        self, data_encoder: SQuADEncoder, tokenizer: BertTokenizerFast, split: str
    ):
        super().__init__(data_encoder, tokenizer, split)

    def __iter__(self):
        for bucket in self.data_encoder.get_buckets():
            for instance in bucket:
                inputs = {
                    "input_ids": instance.input_ids,
                    "token_type_ids": instance.token_type_ids,
                    "attention_mask": instance.attention_mask,
                }
                labels = instance.answer_offsets

                yield inputs, labels

    def collate(
        self, batch: List[SQuADEncoding]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs = self.pad_and_clip_batch(batch)
        # TODO: At the moment this only accepts one answer to be checked in the dev set.
        # This is why the labels are indexed at 0.
        labels = torch.tensor([labels[0] for _, labels in batch]).long()

        return inputs, labels

    def pad_and_clip_batch(
        self, batch: List[SQuADEncoding], model_max_length=MAX_LENGTH
    ) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id

        max_len = max(inp["input_ids"].shape[1] for inp, _ in batch)

        input_ids = []
        token_type_ids = []
        attention_mask = []

        for inp, _ in batch:
            length = inp["input_ids"].shape[1]

            # Pad and clip the tensors by max model length.
            input_ids.append(
                F.pad(
                    inp["input_ids"].squeeze(),
                    (0, max_len - length),
                    value=pad_token_id,
                )[:model_max_length]
            )
            token_type_ids.append(
                F.pad(inp["token_type_ids"].squeeze(), (0, max_len - length), value=1)[:model_max_length]
            )
            attention_mask.append(
                F.pad(inp["attention_mask"].squeeze(), (0, max_len - length), value=1)[:model_max_length]
            )

        input_ids = torch.stack(input_ids)
        token_type_ids = torch.stack(token_type_ids)
        attention_mask = torch.stack(attention_mask)

        inputs = {
            "input_ids": input_ids.long(),
            "token_type_ids": token_type_ids.long(),
            "attention_mask": attention_mask.long(),
        }

        assert input_ids.shape[1] <= model_max_length, f"Input ids have shape {input_ids.shape[1]}, which is too long."

        return inputs
