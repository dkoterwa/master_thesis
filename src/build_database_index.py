from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import faiss
from faiss import write_index, read_index
import numpy as np
from typing import List, Tuple
from utils import TextDataset
from torch.utils.data import DataLoader
import argparse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#we want to take a model, then download and concatenate all of the datasets and embed them, after embedding we pack it into faiss index and store

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
        with torch.no_grad():
            output = self.embedding_model(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          output_hidden_states=True)
            pooled_output = self.pooling(output, attention_mask)
            normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output.numpy()

def run(args: argparse.Namespace) -> None:
    model = Model(args.model_name, args.pooling_type)
    dataset = TextDataset()
    dataset.get_default_data()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    database_embeddings = []
    
    for batch in dataloader:
        output = model.tokenize_and_produce_model_output(batch)
        database_embeddings.extend(output)
        
    embeddings_array = np.array(database_embeddings)
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings_array)
    write_index(index, f"{args.index_output_dir}/{args.model_name.split('/')[1]}.index")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pooling_type", type=str, default="cls")
    parser.add_argument("--index_output_dir", type=str, default="../data/faiss_indexes")
    args = parser.parse_args()
    run(args)
    #TO DO: preprocess datasets forming the database, because now they have different structures

    
    
    
        