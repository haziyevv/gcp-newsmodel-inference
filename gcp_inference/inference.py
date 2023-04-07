
import re
import logging

import torch
from transformers import AutoTokenizer

from .model import AzeNewsModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Inference:
    """
    Class for running inference with a trained AzeNewsModel.
    """

    def __init__(self, model_name, tokenizer_name, max_length):
        """
        Initializes an instance of Inference with a trained model and tokenizer.

        Args:
            model_name (str): Path to the trained model checkpoint.
            tokenizer_name (str): Name of the pre-trained tokenizer.
            max_length (int): Maximum sequence length to use for tokenization.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.model = AzeNewsModel(vocab_size=self.tokenizer.vocab_size, emb_size=128)
        self.model.load_state_dict(torch.load(model_name))
        self.model.eval()

        self.index2cat = {
            0: "iqtisadi",
            1: "medeniyyet",
            2: "siyasi",
            3: "idman",
            4: "ikt",
        }

    def predict(self, instances):
        """
        Runs inference on a list of instances.

        Args:
            instances (list): List of dicts containing 'text' fields.

        Returns:
            list: List of predicted labels as strings.
        """
        instances = [
            re.sub(r"\[.*?\]|[\s\u200b]+", " ", x["text"])
            .replace("  ", "")
            .strip()
            .lower()
            for x in instances
        ]
        inputs = self.tokenizer(
            instances,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            logits = self.model(input_ids)
        return logits.argmax(dim=1)
