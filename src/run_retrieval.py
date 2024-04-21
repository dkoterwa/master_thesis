import pandas as pd
import numpy as np
import faiss
import argparse
import os 
from datasets import load_dataset
from faiss import read_index
from tqdm import tqdm
from utils import DATABASE_PATH
RETRIEVAL_RESULTS_PATH = "../data/retrieval_results"

def find_neighbors_in_index(candidate: np.ndarray, index: faiss.IndexFlat, k: int=1):
    scores, indices = index.search(candidate, k)
    return scores, indices

def run(args):
    database_index = read_index(args.index_path)
    database_df = pd.read_pickle(DATABASE_PATH)
    test_df = load_dataset(args.dataset_name, split="train").to_pandas()
    test_embeddings = np.load(args.test_embeddings_path, allow_pickle=True)
    nearest_neighbors_texts = []
    nearest_neighbors_scores = []
    nearest_neighbor_langs = []
    
    for candidate in tqdm(test_embeddings, desc="Retrieving nearest neighbors from index for test embeddings"):
        scores, indices = find_neighbors_in_index(np.expand_dims(candidate, axis=0), database_index, k=1)
        nearest_neighbor_text = database_df.iloc[indices.item()]["text"]
        nearest_neighbor_lang = database_df.iloc[indices.item()]["lang"]
        nearest_neighbors_texts.append(nearest_neighbor_text)
        nearest_neighbors_scores.append(scores.item())
        nearest_neighbor_langs.append(nearest_neighbor_lang)
        
    test_df["nearest_neighbor_text"] = nearest_neighbors_texts
    test_df["nearest_neighbor_score"] = nearest_neighbors_scores
    test_df["nearest_neighbor_lang"] = nearest_neighbor_langs
    dataset_name = args.dataset_name.split("/")[1]
    model_name = args.model_name.split("/")[1]
    os.makedirs(RETRIEVAL_RESULTS_PATH, exist_ok=True)
    test_df.to_csv(f"{RETRIEVAL_RESULTS_PATH}/{dataset_name}_{model_name}.csv")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="HuggingFace name of the dataset", required=True)
    parser.add_argument("--model_name", help="HuggingFace name of the model", type=str, required=True)
    parser.add_argument("--index_path", type=str, help="Path to the faiss index with the database", required=True)
    parser.add_argument("--test_embeddings_path", type=str)
    args = parser.parse_args()
    run(args)