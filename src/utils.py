from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List
import os
import pandas as pd

#all of these datasets only have train split on HF
DATABASE_DATASETS_MAPPING = {"FreedomIntelligence/alpaca-gpt4-deutsch": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-spanish": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-chinese": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-arabic": "conversations",
                             "5CD-AI/Vietnamese-c-s-ale-alpaca-gpt4-data-gg-translated": "output_vi",
                             "vicgalle/alpaca-gpt4": "output",
                             "MBZUAI/LaMini-instruction": "response",
                             }
DATABASE_PATH = "../data/database.pkl"
class TextDataset(Dataset):
    def __init__(self, texts = None):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
    def get_default_data(self) -> None:
        full_data = []
        for dataset, text_column in DATABASE_DATASETS_MAPPING.items():
            df = load_dataset(dataset, split="train").to_pandas()
            texts = df[text_column].to_list()
            full_data.extend(texts)
        if os.path.isfile(DATABASE_PATH):
            pass
        else:
            database_df = pd.DataFrame({"id": range(1, len(full_data) + 1), "text": full_data})
            database_df.to_pickle(DATABASE_PATH)
        self.texts = full_data
        self.texts = ["test" for i in range(100)]
            
            