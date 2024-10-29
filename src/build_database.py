import pandas as pd
from datasets import load_dataset
import jieba
import os

DATASETS = [
    {"name": "nomic-ai/gpt4all-j-prompt-generations", "answer_column": "response", "lang": "en"},
    {"name": "sahil2801/CodeAlpaca-20k", "answer_column": "output", "lang": "en"},
    {"name": "dkoterwa/ai_society_instructions", "answer_column": "output", "lang": "en"},
    {
        "name": "dkoterwa/camel_ai_biology_instruction_dataset",
        "answer_column": "response",
        "lang": "en",
    },
    {
        "name": "dkoterwa/camel_ai_physics_instruction_dataset",
        "answer_column": "response",
        "lang": "en",
    },
    {
        "name": "dkoterwa/camel_ai_chemistry_instruction_dataset",
        "answer_column": "response",
        "lang": "en",
    },
    {
        "name": "dkoterwa/camel_ai_maths_instruction_dataset",
        "answer_column": "response",
        "lang": "en",
    },
    {"name": "vicgalle/alpaca-gpt4", "answer_column": "output", "lang": "en"},
    {"name": "dkoterwa/alpaca_gpt_4_ar", "answer_column": "answer", "lang": "ar"},
    {"name": "dkoterwa/alpaca_gpt_4_zh", "answer_column": "answer", "lang": "zh"},
    {"name": "dkoterwa/alpaca_gpt_4_es", "answer_column": "answer", "lang": "es"},
    {"name": "dkoterwa/alpaca_gpt_4_de", "answer_column": "answer", "lang": "de"},
    {
        "name": "5CD-AI/Vietnamese-c-s-ale-alpaca-gpt4-data-gg-translated",
        "answer_column": "output_vi",
        "lang": "vi",
    },
    {"name": "MBZUAI/LaMini-instruction", "answer_column": "response", "lang": "en"},
]
DATABASE_PATH = "../data/database.pkl"


def main():
    dfs = []
    for dataset in DATASETS:
        df = load_dataset(dataset["name"], split="train").to_pandas()
        if dataset["name"] == "MBZUAI/LaMini-instruction":
            df = df[
                df["instruction_source"].str.contains("generated")
                | df["instruction_source"].str.contains("self_instruct")
            ]
        texts = df[dataset["answer_column"]]
        temp_df = pd.DataFrame({"text": texts, "lang": dataset["lang"]})
        dfs.append(temp_df)

    full_database = pd.concat(dfs)
    print(f"Size before filtering {len(full_database)}")
    full_database["text_length"] = [
        len(row["text"].split(" ")) if row["lang"] != "zh" else len(jieba.lcut(row["text"]))
        for _, row in full_database.iterrows()
    ]
    full_database = full_database[full_database["text_length"] > 20]
    print(f"Size after filtering {len(full_database)}")
    full_database.drop_duplicates("text", inplace=True)
    full_database.reset_index(drop=True, inplace=True)
    os.makedirs("../data", exist_ok=True)
    full_database.to_pickle(DATABASE_PATH)


if __name__ == "__main__":
    main()
