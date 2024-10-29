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

DATABASE_PATH = "../data/database.pkl"
DATABASE_TEXT_COLUMN = "text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_TO_TEST = [
    "sentence-transformers/distiluse-base-multilingual-cased-v2", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
    "intfloat/multilingual-e5-base",
    "Alibaba-NLP/gte-multilingual-base",
    "BAAI/bge-m3"
    ]

DATASETS_TO_TEST = ["dkoterwa/mkqa_filtered",
                    "dkoterwa/mlqa_filtered",
                    "dkoterwa/oasst2_filtered"]

class TextDataset(Dataset):
    def __init__(self, texts = None):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
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
    
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]
            
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
        self.embedding_model.to(DEVICE)
        self.embedding_model.config.use_flash_attention = False
        self.pooling = Pooling(pooling_type)
    
    def _load_model_and_tokenizer(self) ->Tuple[AutoModel, AutoTokenizer]:
        return AutoModel.from_pretrained(self.hf_name, trust_remote_code=True), AutoTokenizer.from_pretrained(self.hf_name, trust_remote_code=True)
    
    def tokenize_and_produce_model_output(self, data: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(data, 
                                       padding="max_length", 
                                       truncation=True, 
                                       max_length=256, 
                                       return_tensors="pt"
                                       )
        attention_mask = encoded_input["attention_mask"].to(DEVICE)
        input_ids = encoded_input["input_ids"].to(DEVICE)
        
        with torch.no_grad():
            output = self.embedding_model(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          output_hidden_states=True)
            pooled_output = self.pooling(output, attention_mask)
            normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output.cpu().numpy()