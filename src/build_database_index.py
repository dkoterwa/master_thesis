from transformers import AutoModel, AutoTokenizer
import torch
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#all of these datasets only have train split on HF
DATABASE_DATASETS_MAPPING = {"FreedomIntelligence/alpaca-gpt4-deutsch": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-spanish": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-chinese": "conversations",
                             "FreedomIntelligence/alpaca-gpt4-arabic": "conversations",
                             "5CD-AI/Vietnamese-c-s-ale-alpaca-gpt4-data-gg-translated": "output_vi",
                             "vicgalle/alpaca-gpt4": "text",
                             "MBZUAI/LaMini-instruction": "response",
                             }

#we want to take a model, then download and concatenate all of the datasets and embed them, after embedding we pack it into faiss index and store

class Model:
    def __init__(self, hf_name: str,) -> None:
        self.hf_name = hf_name
        self.embedding_model, self.tokenizer = self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        return AutoModel.from_pretrained(self.hf_name), AutoTokenizer.from_pretrained(self.hf_name)
    
    def tokenize_and_produce_model_output(self, data: List[str]):
        encoded_input = self.tokenizer(data, 
                                       padding="max_length", 
                                       truncation=True, 
                                       max_length=128, 
                                       return_tensors="pt"
                                       )
        self.attention_mask = encoded_input["attention_mask"].to(DEVICE)
        input_ids = encoded_input["input_ids"].to(DEVICE)
        with torch.no_grad():
            output = self.embedding_model(input_ids=input_ids, 
                                          attention_mask=self.attention_mask, 
                                          output_hidden_states=True)
            print(output["hidden_states"][0].shape)
    
if __name__ == "__main__":
    model = Model("sentence-transformers/distiluse-base-multilingual-cased-v2")
    test = "I would like to order some pizza"
    model.tokenize_and_produce_model_output(test)
        
        