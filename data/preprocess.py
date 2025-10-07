#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from utils import ERROR, RIGHT, PiscesLxCoreCacheManagerFacade
from datasets import load_from_disk, DatasetDict

def get_subsets_from_model_txt():
    """
    Retrieve dataset subset names from the model.txt file.

    This function constructs the path to the model.txt file, checks if it exists,
    and reads all non-empty lines that do not start with '#' as dataset subset names.

    Returns:
        list: A list of dataset subset names. Returns an empty list if the file does not exist.
    """
    # Use cache manager for model.txt file path
    cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
    data_cache_dir = cache_manager.get_cache_dir("data_cache")
    model_txt_path = os.path.join(data_cache_dir, "model.txt")
    
    # Check if the model.txt file exists
    if not os.path.exists(model_txt_path):
        ERROR(f"{model_txt_path} not found! Please create it with one dataset name per line.")
        return []
    
    # Open the model.txt file and read dataset subset names
    with open(model_txt_path, "r", encoding="utf-8") as file:
        # Filter out empty lines and comment lines, then strip whitespace
        return [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]

def build_splits(subset):
    """
    Build training and test splits for a given dataset subset.

    This function checks if the dataset subset exists. If it does, it attempts to perform a 90/10 train-test split.
    If the dataset doesn't support the train_test_split method, it uses a simple selection for the test set.
    The split dataset is then saved back to disk.

    Args:
        subset (str): The name of the dataset subset.

    Returns:
        None: Returns None if the dataset subset does not exist.
    """
    # Construct the path to the dataset subset
    subset_path = f"data/{subset}"
    
    # Check if the dataset subset exists
    if not os.path.exists(subset_path):
        ERROR(f"{subset_path} does not exist, please run download.py first")
        return
    
    # Load the dataset from disk
    dataset = load_from_disk(subset_path)
    
    # Perform a simple 90/10 split
    if hasattr(dataset, "train_test_split"):
        # Perform a 90/10 train-test split if the dataset supports it
        split_data = dataset.train_test_split(test_size=0.1, seed=42)
    else:
        # If the dataset doesn't support train_test_split, use a simple selection for test set
        split_data = {"train": dataset, "test": dataset.select(range(min(1000, len(dataset))))}
    
    # Convert the split data into a DatasetDict object
    dataset_dict = DatasetDict(split_data)
    
    # Save the split dataset back to disk
    dataset_dict.save_to_disk(subset_path)
    RIGHT(f"{subset} split completed �?{subset_path}")

if __name__ == "__main__":
    # Get all dataset subset names
    SUBSETS = get_subsets_from_model_txt()
    
    # Build splits for each dataset subset
    for subset in SUBSETS:
        build_splits(subset)