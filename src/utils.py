from torch.utils.data import Dataset
from typing import List

#all of these datasets only have train split on HF
DATABASE_DATASETS_MAPPING = {"FreedomIntelligence/alpaca-gpt4-deutsch": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-spanish": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-chinese": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-arabic": "conversations",
                             "5CD-AI/Vietnamese-c-s-ale-alpaca-gpt4-data-gg-translated": "output_vi",
                             "vicgalle/alpaca-gpt4": "text",
                             "MBZUAI/LaMini-instruction": "response",
                             }

class TextDataset(Dataset):
    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]