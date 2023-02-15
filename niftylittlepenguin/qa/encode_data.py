from dataclasses import dataclass
import os
import torch

from abc import ABC, abstractmethod
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from typing import List, Optional
from niftylittlepenguin.qa.read_data import SQuADInstance
from niftylittlepenguin.qa.offset_mapping import QAOffsetMapper
from niftylittlepenguin.qa.constants import MAX_LENGTH
from niftylittlepenguin.qa.config import STORE_ENC
from niftylittlepenguin.shared.utils import create_dirs


class QADataEncoder(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        split: str,
        store_enc: bool = STORE_ENC,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.store_enc = store_enc
        self.encodings = []

    @abstractmethod
    def encodings_exist(self):
        raise NotImplementedError

    @abstractmethod
    def store_encodings(self):
        raise NotImplementedError

    @abstractmethod
    def batch_encode(self, data: list):
        raise NotImplementedError

    @abstractmethod
    def encode(self):
        raise NotImplementedError


@dataclass
class SQuADEncoding:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    # This must be the wordpiece start and end.
    answer_offsets: List[torch.Tensor]


class SQuADEncoder(QADataEncoder):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        split: str,
        store_enc: bool = STORE_ENC,
    ):
        super().__init__(tokenizer, split, store_enc=store_enc)
        self.enc_path = f"encodings/qa/SQuAD-{split}-encodings.pt"

    def encodings_exist(self):
        return os.path.exists(self.enc_path)

    def store_encodings(self):
        with open(self.enc_path, "wb") as f:
            torch.save(self.encodings, f)

    def load_encodings(self):
        with open(self.enc_path, "rb") as f:
            self.encodings = torch.load(f)

    def batch_encode(self, data: List[SQuADInstance]):
        # Invalid encodings (None) are ignored.
        if self.encodings_exist():
            print("Encodings already exist. Loading encodings...")
            self.load_encodings()
        else:
            print("Encoding data...")
            self.encodings = [
                self.encode(squad_instance)
                for squad_instance in tqdm(data)
                if squad_instance is not None
            ]

            if self.store_enc:
                print("Creating 'encodings/qa' directory...")
                create_dirs(self.enc_path)
                print("Storing encodings...")
                self.store_encodings()

        return self.encodings

    def encode(self, squad_instance: SQuADInstance) -> Optional[List[SQuADEncoding]]:
        encoded_question = self.tokenizer(
            squad_instance.question, return_tensors="pt", return_offsets_mapping=True
        )
        encoded_context = self.tokenizer(
            squad_instance.context, return_tensors="pt", return_offsets_mapping=True
        )

        question_len = encoded_question["input_ids"].shape[1]
        context_len = encoded_context["input_ids"].shape[1]

        input_ids = torch.cat(
            [encoded_question["input_ids"], encoded_context["input_ids"][:, 1:]], dim=1
        )
        # Context length is taken -1 because the first token is the [CLS] token.
        token_type_ids = torch.cat(
            [encoded_question["token_type_ids"], torch.ones(1, context_len - 1)], dim=1
        )
        attention_mask = torch.ones_like(input_ids)

        if self.split == "train" and len(squad_instance.answers) > 1:
            print(
                "WARNING: Training QA data should only have one answer but multiple"
                " answers were found."
            )

        mapper = QAOffsetMapper()
        answer_offsets = []

        assert len(squad_instance.answers) == len(
            squad_instance.answer_starts
        ), "The number of answers and answer starts must be the same."

        for answer, answer_start in zip(
            squad_instance.answers, squad_instance.answer_starts
        ):
            # The context offset_mapping is 3-dimensional so the first dimension is removed by indexing.
            context_offsets = mapper.map(
                encoded_context["offset_mapping"][0], answer_start, answer=answer
            )
            # Some answers in the data are incomplete words. The answer is ignored in this case.
            try:
                start, end = mapper.add_question_offsets(context_offsets, question_len)
            except TypeError:
                continue

            # If the answer is outside of the model capacity, it is ignored.
            if end > MAX_LENGTH:
                continue

            answer_offsets.append((start, end))

        # If all answers have been ignored, no encoding is returned.
        if len(answer_offsets) == 0:
            return None

        encoding = SQuADEncoding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            answer_offsets=answer_offsets,
        )

        return encoding
