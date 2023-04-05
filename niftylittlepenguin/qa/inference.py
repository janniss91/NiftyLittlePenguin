import logging
import sys

from transformers import BertTokenizerFast

from niftylittlepenguin.qa.config import BATCH_SIZE, INFERENCE_MODEL
from niftylittlepenguin.qa.constants import MODEL
from niftylittlepenguin.qa.dataset import SQuAD_Dataset
from niftylittlepenguin.qa.encode_data import SQuADEncoder
from niftylittlepenguin.qa.model import LitQABert
from niftylittlepenguin.qa.read_data import SQuADInstance, SQuADReader


def get_inference_model(checkpoint: str):
    model = LitQABert.load_from_checkpoint(checkpoint)
    model.eval()
    return model


def get_inference_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL)
    return tokenizer


def infer_val_dataset(model: LitQABert, tokenizer: BertTokenizerFast):
    dev_datareader = SQuADReader("dev")
    dev_datareader.extract_data()

    for instance in dev_datareader.qa_instances:
        infer(model, tokenizer, instance, verbose=True, with_labels=True)


def infer(
    model: LitQABert,
    tokenizer: BertTokenizerFast,
    input_instance: SQuADInstance,
    verbose: bool = False,
    with_labels: bool = False,
) -> str:
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)

    dev_encoder = SQuADEncoder(tokenizer, "dev")
    instance = dev_encoder.encode(input_instance)
    inputs = {
        "input_ids": instance.input_ids.long(),
        "token_type_ids": instance.token_type_ids.long(),
        "attention_mask": instance.attention_mask.long(),
    }
    labels = instance.answer_offsets

    logits = model(inputs)

    # Separate the predictions for start and end.
    start_logits = logits[:, :, 0]
    end_logits = logits[:, :, 1]

    start_pred = start_logits.argmax(dim=1).item()
    end_pred = end_logits.argmax(dim=1).item()

    context = tokenizer.decode(instance.input_ids[0])
    predicted_answer = tokenizer.decode(
        instance.input_ids[0][start_pred : end_pred + 1]
    )

    if with_labels:
        # Separate the true starts and ends.
        # NOTE: At the moment this only accepts one answer but not multiple.
        # Multiple answers could look like this: [(0, 1)], (2, 3)]
        start, end = labels[0]
        labels = instance.answer_offsets
        answer = tokenizer.decode(instance.input_ids[0][start : end + 1])

    if verbose:
        if with_labels:
            if start_pred == start and end_pred == end:
                print("CORRECT!")
            elif start_pred == start:
                print("Start correct, end wrong!")
            elif end_pred == end:
                print("End correct, start wrong!")
            else:
                print("Both wrong!")

            print("Predictions: ", start_pred, end_pred)
            print("Gold: ", start, end)

        print("Input: ", context)
        print("Predicted answer: ", predicted_answer)

        if with_labels:
            print("Answer: ", answer)
        print()

    return {"answer": predicted_answer, "start_pred": start_pred, "end_pred": end_pred}


if __name__ == "__main__":
    mode = sys.argv[1]

    model = get_inference_model(INFERENCE_MODEL)
    tokenizer = get_inference_tokenizer()

    if mode == "batch":
        infer_val_dataset(model, tokenizer)

    elif mode == "single":
        instance = SQuADInstance(
            title="test squad instance",
            context="This is a test.",
            question="What is this?",
            answer_starts=[8],
            is_impossible=False,
            answers=["a test"],
        )

        infer(model, tokenizer, instance, verbose=True, with_labels=True)

    else:
        raise ValueError("Invalid mode. Must be 'batch' or 'single'")
