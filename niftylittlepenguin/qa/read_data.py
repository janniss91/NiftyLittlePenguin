from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import List, Optional

# TODO: Add config file and put this in there? Or should be constant to prevent different download dir?
SQUAD_DATA_DIR = "data/SQuAD"


class QADataReader(ABC):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        self.qa_instances: list = []

    @abstractmethod
    def read(self):
        raise NotImplementedError
    
    @abstractmethod
    def extract_data(self):
        raise NotImplementedError
    

@dataclass
class SQuADInstance:
    """
    A single instance of a SQuAD question.

    Answer and plausible_answer are optional and should be mutually exclusive.
    """
    title: str
    context: str
    question: str
    answer_starts: List[int]
    is_impossible: bool
    answer: Optional[List[str]] = None
    plausible_answer: Optional[List[str]] = None
    

class SQuADReader(QADataReader):
    def __init__(self, split: str):
        super().__init__(data_dir=SQUAD_DATA_DIR)

        self.split = split
        # Instances that are marked as impossible in the dataset.
        self.imp_instances = []

    def read(self) -> dict:
        path = os.path.join(SQUAD_DATA_DIR, f"{self.split}-v2.0.json")
        with open(path) as f:
            return json.load(f)

    def extract_data(self):
        data = self.read()

        for text in data["data"]:
            title = text["title"]
            for paragraph in text["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    is_impossible = qa["is_impossible"]

                    # The answer set is used to prevent storing duplicates.
                    answer_set = set()
                    # Traverse all answers and ignore duplicates.
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]

                        if (answer_text, answer_start) not in answer_set:
                            # Add the answer to the set of answers.
                            answer_set.add((answer_text, answer_start))

                    if len(qa["answers"]) > 0:
                        qa_instance = SQuADInstance(
                            title=title,
                            context=context,
                            question=question,
                            answer_starts=[answer_start for _, answer_start in answer_set],
                            is_impossible=is_impossible,
                            answer=[answer_text for answer_text, _ in answer_set],
                        )
                        self.qa_instances.append(qa_instance)

                    else:
                        for plausible_answer in qa["plausible_answers"]:
                            answer_text = plausible_answer["text"]
                            answer_start = plausible_answer["answer_start"]

                            if (answer_text, answer_start) not in answer_set:
                                # The answer set can be used for plausible answers, too.
                                answer_set.add((answer_text, answer_start))

                        imp_instance = SQuADInstance(
                            title=title,
                            context=context,
                            question=question,
                            answer_starts=[answer_start for _, answer_start in answer_set],
                            is_impossible=is_impossible,
                            plausible_answer=[answer_text for answer_text, _ in answer_set],
                        )
                        self.imp_instances.append(imp_instance)
