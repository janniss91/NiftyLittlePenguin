import pytorch_lightning as pl
import torch
from memory_profiler import profile
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from niftylittlepenguin.qa.dataset import SQuAD_Dataset
from niftylittlepenguin.qa.encode_data import SQuADEncoder
from niftylittlepenguin.qa.model import LitQABert
from niftylittlepenguin.qa.read_data import SQuADReader
from niftylittlepenguin.shared.download import Downloader

from niftylittlepenguin.qa.constants import MAX_LENGTH, TRAIN_URL
from niftylittlepenguin.qa.constants import DEV_URL
from niftylittlepenguin.qa.constants import TRAIN_PATH
from niftylittlepenguin.qa.constants import DEV_PATH
from niftylittlepenguin.qa.constants import MODEL
from niftylittlepenguin.qa.config import BATCH_SIZE, LOG_INTERVAL, MAX_EPOCHS, TORCH_SEED


@profile
def run():
    # Set torch seed.
    torch.manual_seed(TORCH_SEED)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL)
    train_encoder = SQuADEncoder(tokenizer, "train")
    dev_encoder = SQuADEncoder(tokenizer, "dev")

    if not (train_encoder.buckets_exist() or dev_encoder.buckets_exist()):
        # Download the train and dev data.
        downloader = Downloader()
        downloader.download(TRAIN_URL, TRAIN_PATH)
        downloader.download(DEV_URL, DEV_PATH)

        # Read the train and dev data.
        train_datareader = SQuADReader("train")
        dev_datareader = SQuADReader("dev")

        # NOTE: There are impossible instances in the datareaders that can potentially added if needed or for experimental purposes.
        train_datareader.extract_data()
        dev_datareader.extract_data()

        train_encoder.batch_encode(train_datareader.qa_instances)
        dev_encoder.batch_encode(dev_datareader.qa_instances)

    train_dataset = SQuAD_Dataset(train_encoder, tokenizer, "train")
    dev_dataset = SQuAD_Dataset(dev_encoder, tokenizer, "dev")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate
    )
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=dev_dataset.collate)

    qa_model = LitQABert()
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(log_every_n_steps=LOG_INTERVAL, max_epochs=MAX_EPOCHS, accelerator=device)
    trainer.fit(
        model=qa_model, train_dataloaders=train_loader, val_dataloaders=dev_loader
    )

if __name__ == "__main__":
    run()
