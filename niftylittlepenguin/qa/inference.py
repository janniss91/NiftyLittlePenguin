import sys
from transformers import BertTokenizerFast
from niftylittlepenguin.qa.config import BATCH_SIZE

from niftylittlepenguin.qa.dataset import SQuAD_Dataset
from niftylittlepenguin.qa.encode_data import SQuADEncoder
from niftylittlepenguin.qa.model import LitQABert

from niftylittlepenguin.qa.constants import MODEL
from niftylittlepenguin.qa.read_data import SQuADInstance, SQuADReader

import logging


def get_inference_model(checkpoint: str):
    model =  LitQABert.load_from_checkpoint(checkpoint)
    model.eval()
    return model


def get_inference_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL)
    return tokenizer


def infer_val_dataset(model: LitQABert, tokenizer: BertTokenizerFast):
    dev_datareader = SQuADReader("dev")
    dev_datareader.extract_data()

    for instance in dev_datareader.qa_instances:
        infer(model, tokenizer, instance)



def infer(model: LitQABert, tokenizer: BertTokenizerFast, input_instance: SQuADInstance):
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

    # Separate the true starts and ends.
    # NOTE: At the moment this only accepts one answer but not multiple.
    # Multiple answers could look like this: [(0, 1)], (2, 3)]
    start, end = labels[0]

    start_pred = start_logits.argmax(dim=1).item()
    end_pred = end_logits.argmax(dim=1).item()

    if start_pred == start and end_pred == end:
        print("CORRECT!")
    elif start_pred == start:
        print("Start correct, end wrong!")
    elif end_pred == end:
        print("End correct, start wrong!")

    print("Predictions: ", start_pred, end_pred)
    print("Gold: ", start, end)

    print("Input: ", tokenizer.decode(instance.input_ids[0]))
    # Plus 1 must be added because the end is inclusive.
    print("Answer: ", tokenizer.decode(instance.input_ids[0][start:end + 1]))
    print("Predicted answer: ", tokenizer.decode(instance.input_ids[0][start_pred:end_pred + 1]))
    print()


if __name__ == "__main__":
    mode = sys.argv[1]

    model = get_inference_model("lightning_logs/version_32/checkpoints/epoch=2-step=25980.ckpt")
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
        
        infer(model, tokenizer, instance)

    else:
        raise ValueError("Invalid mode. Must be 'batch' or 'single'")
