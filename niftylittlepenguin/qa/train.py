from transformers import BertTokenizerFast
from niftylittlepenguin.qa.dataset import SQuAD_Dataset
from niftylittlepenguin.qa.encode_data import SQuADEncoder
from niftylittlepenguin.qa.read_data import SQuADReader
from niftylittlepenguin.shared.download import Downloader

from niftylittlepenguin.qa.constants import TRAIN_URL
from niftylittlepenguin.qa.constants import DEV_URL
from niftylittlepenguin.qa.constants import TRAIN_PATH
from niftylittlepenguin.qa.constants import DEV_PATH
from niftylittlepenguin.qa.constants import MODEL


def run():
    # Download the train and dev data.
    downloader = Downloader()
    downloader.download(TRAIN_URL, TRAIN_PATH)
    downloader.download(DEV_URL, DEV_PATH)

    # Read the train and dev data.
    train_datareader = SQuADReader("train")
    dev_datareader = SQuADReader("dev")
    train_datareader.extract_data()
    dev_datareader.extract_data()
    train_data = train_datareader.qa_instances
    dev_data = dev_datareader.qa_instances
    # NOTE: There are impossible instances that can potentially added if needed or for experimental purposes.
    # train_imp_data = train_datareader.imp_instances
    # dev_imp_data = dev_datareader.imp_instances

    tokenizer = BertTokenizerFast.from_pretrained(MODEL)

    train_encoder = SQuADEncoder(tokenizer, "train")
    dev_encoder = SQuADEncoder(tokenizer, "dev")

    train_samples = train_encoder.batch_encode(train_data)
    dev_samples = dev_encoder.batch_encode(dev_data)

    train_dataset = SQuAD_Dataset(train_samples, "train")
    dev_dataset = SQuAD_Dataset(dev_samples, "dev")

    print(len(train_dataset))
    print(len(dev_dataset))


if __name__ == "__main__":
    run()
