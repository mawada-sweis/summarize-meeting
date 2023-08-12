from datasets import load_dataset
import pandas as pd
import os

DATASET_FILE_PATH = "./dataset/transcript_summary.csv"


def is_dataset_loaded():
    return os.path.exists(DATASET_FILE_PATH)


def load_dataset():
    dataset = load_dataset('TalTechNLP/AMIsum',
                           split='train+validation+test').to_pandas()

    dataset = pd.DataFrame(dataset.drop(columns='id'))

    dataset.to_csv(DATASET_FILE_PATH, index=False)

    print("You have install TalTechNLP/AMIsum dataset successfully.")
    print(f"Dataset shape: {dataset.shape}\n \
          Dataset columns: {dataset.columns}")
