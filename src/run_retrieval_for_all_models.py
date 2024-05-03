import logging
import pandas as pd
import numpy as np
import faiss
import os 
from datasets import load_dataset
from faiss import read_index
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import DATABASE_PATH, EmbeddingsDataset
logging.basicConfig(filename="retrieval.log", filemode="w", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RETRIEVAL_RESULTS_PATH = "../data/retrieval_results"

MODELS_TO_TEST = ["distiluse-base-multilingual-cased-v2",
                  "paraphrase-multilingual-MiniLM-L12-v2", 
                  "paraphrase-multilingual-mpnet-base-v2", 
                  "LaBSE",
                  "bert-base-multilingual-cased",
                  "xlm-roberta-base"]

DATASETS_TO_TEST = ["dkoterwa/mkqa_filtered",
                    "dkoterwa/mlqa_filtered",
                    "dkoterwa/oasst2_filtered"]

class args:
    def __init__(self, model_name, dataset_name, index_path, test_embeddings_path):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.index_path = index_path
        self.test_embeddings_path = test_embeddings_path
        
def find_neighbors_in_index(candidate: np.ndarray, index: faiss.IndexFlat, k: int=1):
    scores, indices = index.search(candidate, k)
    return scores, indices

def run(args):
    database_index = read_index(args.index_path)
    logging.info("Index loaded")
    database_df = pd.read_pickle(DATABASE_PATH)
    logging.info("Database loaded")
    test_df = load_dataset(args.dataset_name, split="train").to_pandas()
    logging.info("Test dataset loaded")
    test_embeddings = np.load(args.test_embeddings_path, allow_pickle=True)
    dataset = EmbeddingsDataset(test_embeddings)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    logging.info("Test embeddings loaded")
    nearest_neighbors_texts = []
    nearest_neighbors_scores = []
    nearest_neighbor_langs = []

    logging.info(f"Running retrieval for {args.model_name} model and {args.dataset_name.split('/')[1]} dataset")    
    for batch in tqdm(dataloader ,desc="Retrieving nearest neighbors from index for test embeddings"):
        scores, indices = find_neighbors_in_index(batch.numpy(), database_index, k=1)
        flat_indices = [item for sublist in indices for item in sublist]
        flat_scores = [item for sublist in scores for item in sublist]
        nearest_neighbor_text = database_df.iloc[flat_indices]["text"]
        nearest_neighbor_lang = database_df.iloc[flat_indices]["lang"]
        nearest_neighbors_texts.extend(nearest_neighbor_text.tolist())
        nearest_neighbors_scores.extend(flat_scores)
        nearest_neighbor_langs.extend(nearest_neighbor_lang.tolist())
    
    logging.info("Finished retrieving neighbors from index")        
    test_df["nearest_neighbor_text"] = nearest_neighbor_text
    test_df["nearest_neighbor_score"] = flat_scores
    test_df["nearest_neighbor_lang"] = nearest_neighbor_lang     
    
    dataset_name = args.dataset_name.split("/")[1]
    model_name = args.model_name.split("/")[1]
    os.makedirs(RETRIEVAL_RESULTS_PATH, exist_ok=True)
    test_df.to_csv(f"{RETRIEVAL_RESULTS_PATH}/{dataset_name}_{model_name}.csv")
    logging.info(f"Saved CSV file with results for {model_name} model and {dataset_name} dataset")
    
if __name__ == "__main__":
    for model_name in MODELS_TO_TEST:
        for dataset_name in DATASETS_TO_TEST:
            index_path = os.path.join("../data/faiss_indexes/", f"{model_name}.index")
            test_embeddings_path = os.path.join("../data/test_embeddings/", f"{dataset_name.split('/')[1]}_{model_name}.npy")
            arguments = args(model_name=model_name,
                             dataset_name=dataset_name, 
                             index_path=index_path, 
                             test_embeddings_path=test_embeddings_path)
            run(arguments)
