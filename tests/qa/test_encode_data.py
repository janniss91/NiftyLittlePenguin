from typing import List

import pytest
import torch
from transformers import BertTokenizerFast

from niftylittlepenguin.qa.encode_data import SQuADEncoder
from niftylittlepenguin.qa.read_data import SQuADInstance
from niftylittlepenguin.qa.train import MODEL


class TestCharToWordpieceOffsetMapper:
    @pytest.fixture
    def data_single_answer(self) -> List[SQuADInstance]:
        return [
            SQuADInstance(
                title="test squad instance",
                context="This is a test.",
                question="What is this?",
                answer_starts=[8],
                is_impossible=False,
                answers=["a test"],
            ),
        ]

    @pytest.fixture
    def data_mult_answers(self) -> List[SQuADInstance]:
        return [
            SQuADInstance(
                title="test squad instance",
                context="This is a test.",
                question="What is this?",
                answer_starts=[8, 10],
                is_impossible=False,
                answers=["a test", "test"],
            ),
        ]

    @pytest.fixture
    def train_squad_encoder(self) -> SQuADEncoder:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL)
        return SQuADEncoder(tokenizer, "train")

    @pytest.fixture
    def dev_squad_encoder(self) -> SQuADEncoder:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL)
        return SQuADEncoder(tokenizer, "dev")

    def test_encode_single_answer(
        self, train_squad_encoder: SQuADEncoder, data_single_answer: List[SQuADInstance]
    ):
        encoding = train_squad_encoder.encode(data_single_answer[0])

        input_ids = torch.tensor(
            [101, 2054, 2003, 2023, 1029, 102, 2023, 2003, 1037, 3231, 1012, 102]
        )
        token_type_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        attention_mask = torch.tensor([1] * 12)
        answer_offsets = [(8, 9)]

        assert torch.all(encoding.input_ids == input_ids)
        assert torch.all(encoding.token_type_ids == token_type_ids)
        assert torch.all(encoding.attention_mask == attention_mask)
        assert encoding.answer_offsets == answer_offsets

    def test_encode_mult_answers(
        self, dev_squad_encoder: SQuADEncoder, data_mult_answers: List[SQuADInstance]
    ):
        encoding = dev_squad_encoder.encode(data_mult_answers[0])

        input_ids = torch.tensor(
            [101, 2054, 2003, 2023, 1029, 102, 2023, 2003, 1037, 3231, 1012, 102]
        )
        token_type_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        attention_mask = torch.tensor([1] * 12)
        answer_offsets = [(8, 9), (9, 9)]

        assert torch.all(encoding.input_ids == input_ids)
        assert torch.all(encoding.token_type_ids == token_type_ids)
        assert torch.all(encoding.attention_mask == attention_mask)
        assert encoding.answer_offsets == answer_offsets
