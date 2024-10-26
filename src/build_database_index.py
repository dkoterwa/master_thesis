import faiss
import argparse
from faiss import write_index, read_index
import numpy as np
from utils import TextDataset, Model, MODELS_TO_TEST
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

BATCH_SIZE=2048

class arguments:
    def __init__(self, model_name, pooling_type, index_output_dir):
        self.model_name = model_name
        self.pooling_type = pooling_type
        self.index_output_dir = index_output_dir
        
def run(args: argparse.Namespace) -> None:
    model = Model(args.model_name, args.pooling_type)
    dataset = TextDataset()
    dataset.build_database()
    print(f"Size of the database: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    database_embeddings = []
    
    for batch in tqdm(dataloader, desc="Calculating embeddings of the database observations"):
        output = model.tokenize_and_produce_model_output(batch)
        database_embeddings.extend(output)
        
    embeddings_array = np.array(database_embeddings)
    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings_array)
    os.makedirs(args.index_output_dir, exist_ok=True)
    write_index(index, f"{args.index_output_dir}/{args.model_name.split('/')[1]}.index")
    
    
if __name__ == "__main__":
    for model in MODELS_TO_TEST:
        print(f"building for model {model}")
        args = arguments(model_name=model, pooling_type="cls", index_output_dir="../data/faiss_indexes")   
        run(args)

    
    
    
        