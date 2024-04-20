from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List
import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Tuple

#all of these datasets only have train split on HF
DATABASE_DATASETS_MAPPING = {"dkoterwa/alpaca_gpt_4_de": {"answer_column_name": "answer", "lang": "de"},
                             "dkoterwa/alpaca_gpt_4_es": {"answer_column_name": "answer", "lang": "es"},
                             "dkoterwa/alpaca_gpt_4_ar": {"answer_column_name": "answer", "lang": "ar"},
                             "dkoterwa/alpaca_gpt_4_zh": {"answer_column_name": "answer", "lang": "zh"},
                             "5CD-AI/Vietnamese-c-s-ale-alpaca-gpt4-data-gg-translated": {"answer_column_name": "output_vi", "lang": "vi"},
                             "vicgalle/alpaca-gpt4": {"answer_column_name": "output", "lang": "en"},
                             "MBZUAI/LaMini-instruction": {"answer_column_name": "response", "lang": "en"},
                             }
DATABASE_PATH = "../data/database.pkl"
DATABASE_TEXT_COLUMN = "text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    def __init__(self, texts = None):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
    def _get_default_data(self) -> pd.DataFrame:
        df_list = []
        total_num_of_observations = 0
        for dataset, properties in DATABASE_DATASETS_MAPPING.items():
            df = load_dataset(dataset, split="train").to_pandas() #all default datasets only have train split
            texts = df[properties["answer_column_name"]].to_list()
            lang = properties["lang"]
            total_num_of_observations += len(texts)
            temp_df = pd.DataFrame({"text": texts, "lang": lang})
            df_list.append(temp_df)
            
        full_data = pd.concat(df_list, axis=0)
        print(len(full_data), total_num_of_observations)
        full_data.reset_index(drop=True, inplace=True)
        assert total_num_of_observations == len(full_data), "Not all of the observations from datasets picked for database have been correctly added to the final DataFrame"
        self.texts = full_data["text"].to_list()
        return full_data

    def build_database(self) -> None:
        if os.path.isfile(DATABASE_PATH):
            print("Database already exists in the default directory, loading...")
            database = pd.read_pickle(DATABASE_PATH)
            self.texts = database[DATABASE_TEXT_COLUMN]
        else:
            print("Building database")
            database_df = self._get_default_data()  
            database_df["id"] = range(0, len(database_df))
            database_df.to_pickle(DATABASE_PATH)
            
class Pooling:
    def __init__(self, pooling_type: str) -> None:
        self.pooling_type = pooling_type

    def __call__(self, model_outputs, attention_mask=None):
        if self.pooling_type == "cls":
            return model_outputs["last_hidden_state"][:, 0, :]
        elif self.pooling_type == "mean":
            token_embeddings = model_outputs["last_hidden_state"] 
            input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()) #how does expand append values in the newly created dimension? Read more about it
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            raise ValueError("Wrong pooling method provided in the Pooling initialization")

class Model:
    def __init__(self, hf_name: str, pooling_type: str) -> None:
        self.hf_name = hf_name
        self.embedding_model, self.tokenizer = self._load_model_and_tokenizer()
        self.pooling = Pooling(pooling_type)
    
    def _load_model_and_tokenizer(self) ->Tuple[AutoModel, AutoTokenizer]:
        return AutoModel.from_pretrained(self.hf_name), AutoTokenizer.from_pretrained(self.hf_name)
    
    def tokenize_and_produce_model_output(self, data: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(data, 
                                       padding="max_length", 
                                       truncation=True, 
                                       max_length=256, 
                                       return_tensors="pt"
                                       )
        attention_mask = encoded_input["attention_mask"].to(DEVICE)
        input_ids = encoded_input["input_ids"].to(DEVICE)
        self.embedding_model.to(DEVICE)
        with torch.no_grad():
            output = self.embedding_model(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          output_hidden_states=True)
            pooled_output = self.pooling(output, attention_mask)
            normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output.cpu().numpy()