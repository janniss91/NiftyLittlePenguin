import pytest
from transformers import BertTokenizerFast

from niftylittlepenguin.qa.offset_mapping import QAOffsetMapper
from niftylittlepenguin.qa.train import MODEL


class TestCharToWordpieceOffsetMapper:
    @pytest.fixture
    def mapper(self) -> QAOffsetMapper:
        return QAOffsetMapper()

    @pytest.fixture
    def tokenizer(self) -> BertTokenizerFast:
        return BertTokenizerFast.from_pretrained(MODEL)

    def test_add_question_offsets(
        self, mapper: QAOffsetMapper, tokenizer: BertTokenizerFast
    ):
        context = (
            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were"
            " the people who in the 10th and 11th centuries gave their name to"
            ' Normandy, a region in France. They were descended from Norse ("Norman"'
            ' comes from "Norseman") raiders and pirates from Denmark, Iceland and'
            " Norway who, under their leader Rollo, agreed to swear fealty to King"
            " Charles III of West Francia. Through generations of assimilation and"
            " mixing with the native Frankish and Roman-Gaulish populations, their"
            " descendants would gradually merge with the Carolingian-based cultures of"
            " West Francia. The distinct cultural and ethnic identity of the Normans"
            " emerged initially in the first half of the 10th century, and it continued"
            " to evolve over the succeeding centuries."
        )
        question = "What century did the Normans first gain their separate identity?"

        answer = "10th century"
        char_start = 671

        encoded_context = tokenizer(
            context, return_tensors="pt", return_offsets_mapping=True
        )
        encoded_question = tokenizer(
            question, return_tensors="pt", return_offsets_mapping=True
        )

        wordpiece_offsets = encoded_context["offset_mapping"][0]
        question_len = len(encoded_question["input_ids"][0])

        # Offsets are (144, 145)
        context_offsets = mapper.map(wordpiece_offsets, char_start, answer=answer)

        actual = mapper.add_question_offsets(context_offsets, question_len)
        expected = (157, 158)

        assert actual == expected
