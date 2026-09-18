"""
data_loader.py

Purpose:
--------
This file is responsible for loading the retail sales dataset.

Author: Sneha
Project: Retail Sales Analytics Dashboard
"""

import pandas as pd
import os


def load_data(file_path):
    """
    Loads a CSV file and returns a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame
        Loaded dataset.
    """

    # Check whether file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read CSV
    df = pd.read_csv(file_path, encoding="latin1")

    return df


if __name__ == "__main__":

    path = "data/raw/SampleSuperstore.csv"

    df = load_data(path)

    print("Dataset Loaded Successfully!")
    print(df.head())