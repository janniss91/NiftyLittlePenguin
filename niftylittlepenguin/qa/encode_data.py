import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from niftylittlepenguin.qa.config import STORE_ENC
from niftylittlepenguin.qa.constants import MAX_LENGTH
from niftylittlepenguin.qa.offset_mapping import QAOffsetMapper
from niftylittlepenguin.qa.read_data import SQuADInstance
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
    def load_encodings(self):
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
        store_buckets: bool = True,
        bucket_size: int = 1000,
    ):
        # TODO: Remove possibility of storing all data in one file.
        super().__init__(tokenizer, split, store_enc=store_enc)
        self.store_buckets = store_buckets
        self.enc_path = f"encodings/qa/SQuAD-{split}-encodings.pt"
        self.enc_bucket_dir = f"encodings/qa/SQuAD-{split}"
        self.pt_ending = ".pt"
        self.encoding_buckets = []
        self.bucket_size = bucket_size

    def encodings_exist(self):
        return os.path.exists(self.enc_path)
    
    def buckets_exist(self):
        return os.path.exists(self.enc_bucket_dir) and len(os.listdir(self.enc_bucket_dir)) > 0

    def store_encodings(self):
        with open(self.enc_path, "wb") as f:
            torch.save(self.encodings, f)

    def load_encodings(self):
        print("Loading encodings...")
        with open(self.enc_path, "rb") as f:
            self.encodings = torch.load(f)

    def split_encodings(self):
        if self.encodings:
            self.encoding_buckets = [
                self.encodings[i : i + self.bucket_size]
                for i in range(0, len(self.encodings), self.bucket_size)
            ]
        else:
            raise ValueError("Encodings must be loaded before splitting into buckets.")

    def store_encoding_buckets(self):
        print("Storing encoding buckets...")
        for num, encoding_bucket in enumerate(self.encoding_buckets):
            bucket_path = os.path.join(self.enc_bucket_dir, str(num) + self.pt_ending)
            with open(bucket_path, "wb") as f:
                torch.save(encoding_bucket, f)

    def load_encoding_bucket(self, num: int) -> List[SQuADEncoding]:
        bucket_path = os.path.join(self.enc_bucket_dir, str(num) + self.pt_ending)
        with open(bucket_path, "rb") as f:
            return torch.load(f)

    def get_num_buckets(self) -> int:
        return len(os.listdir(self.enc_bucket_dir))
    
    def get_buckets(self, max_buckets: int = 1_000_000) -> Iterator[List[SQuADEncoding]]:
        """
        Max buckets is used to limit the number of buckets that are loaded.
        The default is 1_000_000, which is always higher than the true number of buckets.
        """
        for num in range(min(max_buckets, self.get_num_buckets())):
            yield self.load_encoding_bucket(num)

    def batch_encode(self, data: List[SQuADInstance]):
        print("Encoding data...")
        self.encodings = [
            self.encode(squad_instance)
            for squad_instance in tqdm(data)
        ]
        # Remove None values.
        self.encodings = [encoding for encoding in self.encodings if encoding is not None]

        if self.store_enc:
            if self.store_buckets:
                self.split_encodings()
                print(f"Creating '{os.path.dirname(self.enc_bucket_dir)}' directory...")
                create_dirs(self.enc_bucket_dir + "/")
                self.store_encoding_buckets()
            else:
                print(f"Creating '{os.path.dirname(self.enc_path)}' directory...")
                create_dirs(self.enc_path)
                print("Storing encodings...")
                self.store_encodings()

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

            # If the answer is outside of the model capacity, the sample is ignored.
            if end >= MAX_LENGTH:
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
