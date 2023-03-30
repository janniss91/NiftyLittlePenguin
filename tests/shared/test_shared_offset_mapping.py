import pytest
from transformers import BertTokenizerFast

from niftylittlepenguin.qa.train import MODEL
from niftylittlepenguin.shared.offset_mapping import \
    CharToWordpieceOffsetMapper


class TestCharToWordpieceOffsetMapper:
    @pytest.fixture
    def mapper(self) -> CharToWordpieceOffsetMapper:
        return CharToWordpieceOffsetMapper()

    @pytest.fixture
    def tokenizer(self) -> BertTokenizerFast:
        return BertTokenizerFast.from_pretrained(MODEL)

    def test_map_start_and_answer(
        self, mapper: CharToWordpieceOffsetMapper, tokenizer: BertTokenizerFast
    ):
        context = "This is a test."
        answer = "a test"
        char_start = 8
        # Note: Char end is 8 + 6 = 14

        encoded = tokenizer(context, return_tensors="pt", return_offsets_mapping=True)
        wordpiece_offsets = encoded["offset_mapping"][0]

        actual = mapper.map(wordpiece_offsets, char_start, answer=answer)
        expected = (3, 4)

        assert actual == expected

    def test_map_start_and_end(
        self, mapper: CharToWordpieceOffsetMapper, tokenizer: BertTokenizerFast
    ):
        context = "This is a test."
        char_start = 8
        char_end = 14

        encoded = tokenizer(context, return_tensors="pt", return_offsets_mapping=True)
        wordpiece_offsets = encoded["offset_mapping"][0]

        actual = mapper.map(wordpiece_offsets, char_start, char_end=char_end)
        expected = (3, 4)

        assert actual == expected

    def test_map_start_and_answer_real_example(
        self, mapper: CharToWordpieceOffsetMapper, tokenizer: BertTokenizerFast
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

        answer = "10th century"
        char_start = 671

        encoded = tokenizer(context, return_tensors="pt", return_offsets_mapping=True)
        wordpiece_offsets = encoded["offset_mapping"][0]

        actual = mapper.map(wordpiece_offsets, char_start, answer=answer)
        expected = (144, 145)

        assert actual == expected
