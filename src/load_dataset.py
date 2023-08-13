import os
from typing import Union

import pandas as pd
from datasets import load_dataset, Dataset

folder_path = './dataset/'


def is_dataset_loaded(file_name: str) -> bool:
    """Check if a dataset file exists at the dataset folder.

    Args:
        file_name (str): The name of the dataset file.

    Returns:
        bool: True if the dataset file exists, False otherwise.
    """
    return os.path.exists(folder_path+file_name)


def load_dataset_by_name(dataset_name: str = 'TalTechNLP/AMIsum',
                         split_type: str = 'train+valedation+test'
                         ) -> Dataset:
    """Load a dataset using its name and split type.

    Args:
        dataset_name (str, optional): The name of the dataset.
            Defaults to 'TalTechNLP/AMIsum'.
        split_type (str, optional): The type of split to load. Can be 'train',
            'validation', 'test' or combination of them.
            Defaults to 'train+validation+test'.
    Returns:
        Dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name,
                           split=split_type)

    print(f"You have install {dataset_name} dataset successfully.")
    print(f"Dataset shape: {dataset.shape}\n \
          Dataset columns: {dataset.column_names}")
    return dataset


def save_to_csv(dataset: Union[Dataset, pd.DataFrame],
                dataset_name: str, columns_name: list) -> None:
    """Save a dataset or DataFrame to a CSV file.

    Args:
        dataset (Dataset or pd.DataFrame): The dataset or DataFrame to save.
        dataset_name (str): The name of the output CSV file.
        columns_name (list): List of columns to be saved.
    """
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)
    dataset[columns_name].to_csv(f'{folder_path}{dataset_name}',
                                 index=False)
    print(f"You have save {dataset_name} as csv file successfully.")


def get_loaded_data(file_name: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        file_name (str): The name of the CSV file to load.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    print(f"You have load {file_name} file successfully.")
    return pd.read_csv(f'{folder_path}{file_name}')


def get_tags(text: list) -> list:
    """Extract tags from a list of strings.

    Args:
        text (list): List of strings to extract tags from.

    Returns:
        list: List of extracted tags.
    """
    tags = set(text.str.findall(r'<(.*?)>').sum())
    return [f'<{tag}>' for tag in tags]
