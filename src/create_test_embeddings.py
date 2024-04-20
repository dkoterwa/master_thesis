import numpy as np
import pandas as pd
import argparse
import os
from torch.utils.data import DataLoader
from utils import TextDataset, Model
from typing import List
from tqdm import tqdm
from datasets import load_dataset
EMBEDDINGS_SAVE_PATH = "../data/test_embeddings"

def save_embeddings(embeddings_array: List[np.ndarray], dataset_name:str, model_name: str, output_dir: str=EMBEDDINGS_SAVE_PATH) -> None:
    dataset_name = dataset_name.split("/")[1]
    model_name = model_name.split("/")[1]
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{EMBEDDINGS_SAVE_PATH}/{dataset_name}_{model_name}.npy", embeddings_array)
    
def run(args: argparse.ArgumentParser) -> None:
    model = Model(args.model_name, args.pooling_type)
    data = load_dataset(args.dataset_name, split="train").to_pandas()
    dataset = TextDataset(data[args.text_column_name])
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    test_embeddings = []
    
    for batch in tqdm(dataloader, desc=f"Calculating embeddings of {args.dataset_name} test dataset"):
        output = model.tokenize_and_produce_model_output(batch)
        test_embeddings.extend(output)
        
    embeddings_array = np.array(test_embeddings)
    save_embeddings(embeddings_array, args.dataset_name, args.model_name, args.embeddings_output_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="HuggingFace name of the dataset", required=True)
    parser.add_argument("--model_name", help="HuggingFace name of the model", type=str, required=True)
    parser.add_argument("--pooling_type", type=str, default="cls")
    parser.add_argument("--text_column_name", type=str, default="answer")
    parser.add_argument("--embeddings_output_dir", type=str, default=EMBEDDINGS_SAVE_PATH)
    args = parser.parse_args()
    run(args)