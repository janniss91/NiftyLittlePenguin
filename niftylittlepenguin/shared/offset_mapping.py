import torch
from typing import Optional, Tuple


class CharToWordpieceOffsetMapper:
    def map(
        self,
        wordpiece_offsets: torch.Tensor,
        char_start: int,
        char_end: Optional[int] = None,
        answer: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Convert character offsets to wordpiece offsets.

        The start offset must be given.
        The end offset can either be specified directly or calculated using the answer.
        """
        if char_end is not None:
            return self._map_start_and_end(wordpiece_offsets, char_start, char_end)
        elif answer is not None:
            return self._map_start_and_answer(wordpiece_offsets, char_start, answer)

    def _map_start_and_end(
        self, wordpiece_offsets: torch.Tensor, char_start: int, char_end: int
    ) -> Tuple[int, int]:
        wordpiece_start = None
        wordpiece_end = None

        for wordpiece_num, offset in enumerate(wordpiece_offsets):
            if offset[0] == char_start:
                wordpiece_start = wordpiece_num
            if offset[1] == char_end:
                wordpiece_end = wordpiece_num

        return wordpiece_start, wordpiece_end

    def _map_start_and_answer(
        self, wordpiece_offsets: torch.Tensor, char_start: int, answer: str
    ) -> Tuple[int, int]:
        char_end = char_start + len(answer)
        return self._map_start_and_end(wordpiece_offsets, char_start, char_end)
