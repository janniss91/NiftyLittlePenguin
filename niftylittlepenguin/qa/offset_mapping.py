from typing import Tuple

from niftylittlepenguin.shared.offset_mapping import \
    CharToWordpieceOffsetMapper


class QAOffsetMapper(CharToWordpieceOffsetMapper):
    def add_question_offsets(
        self, context_offsets: Tuple[int, int], question_len: int
    ) -> Tuple[int, int]:
        # Calculate - 1 because the [CLS] token is removed.
        return (
            context_offsets[0] + question_len - 1,
            context_offsets[1] + question_len - 1,
        )
