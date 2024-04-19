import faiss
import argparse
from faiss import write_index, read_index
import numpy as np
from utils import TextDataset, Model
from torch.utils.data import DataLoader
from tqdm import tqdm

def run(args: argparse.Namespace) -> None:
    model = Model(args.model_name, args.pooling_type)
    dataset = TextDataset()
    dataset.build_database()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    database_embeddings = []
    
    for batch in tqdm(dataloader, desc="Calculating embeddings of the database observations"):
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

    
    
    
        