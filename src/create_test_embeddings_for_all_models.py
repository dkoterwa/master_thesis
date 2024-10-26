import numpy as np
import os
from torch.utils.data import DataLoader
from utils import TextDataset, Model, DATASETS_TO_TEST, MODELS_TO_TEST
from typing import List
from tqdm import tqdm
from datasets import load_dataset

# Constants
EMBEDDINGS_SAVE_PATH = "../data/test_embeddings"
POOLING_TYPE = "cls"
TEXT_COLUMN_NAME = "answer"
LANGUAGE_COLUMN_NAME = "lang"
BATCH_SIZE = 2048


def save_embeddings(embeddings_array: List[np.ndarray], dataset_name: str, model_name: str, output_dir: str = EMBEDDINGS_SAVE_PATH) -> None:
    dataset_name = dataset_name.split("/")[1]
    model_name = model_name.split("/")[1]
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/{dataset_name}_{model_name}.npy", embeddings_array)

def run(model_name: str, dataset_name: str) -> None:
    model = Model(model_name, POOLING_TYPE)
    data = load_dataset(dataset_name, split="train").to_pandas()
    dataset = TextDataset(data[TEXT_COLUMN_NAME])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_embeddings = []
    
    for batch in tqdm(dataloader, desc=f"Calculating embeddings of {dataset_name} with {model_name}"):
        output = model.tokenize_and_produce_model_output(batch)
        test_embeddings.extend(output)
        
    embeddings_array = np.array(test_embeddings)
    save_embeddings(embeddings_array, dataset_name, model_name, EMBEDDINGS_SAVE_PATH)

if __name__ == "__main__":
    for dataset_name in DATASETS_TO_TEST:
        for model_name in MODELS_TO_TEST:
            run(model_name, dataset_name)
