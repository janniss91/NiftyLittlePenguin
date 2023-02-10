
from niftylittlepenguin.qa.read_data import SQuADReader
from niftylittlepenguin.shared.download import Downloader

# TODO: Add a config file and put these in there.
TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"

# TODO: Add config file and put this in there? Or should be constant to prevent different download dir?
TRAIN_PATH = "data/SQuAD/train-v2.0.json"
DEV_PATH = "data/SQuAD/dev-v2.0.json"

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
    # train_plausible_data = train_datareader.imp_instances
    # dev_plausible_data = dev_datareader.imp_instances

if __name__ == "__main__":
    run()