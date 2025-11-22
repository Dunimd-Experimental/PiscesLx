#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from utils import PiscesLxCoreCacheManagerFacade
from datasets import load_from_disk, DatasetDict

class PiscesLxToolsDataPreprocessor:
    """Preprocessor for dataset subsets and splits management."""
    
    @staticmethod
    def get_subsets_from_model_txt():
        """
        Retrieves dataset subset names from the model.txt file.

        This function constructs the path to the model.txt file using the cache manager.
        If the file exists, it reads all non-empty lines that do not start with '#' 
        as dataset subset names. Otherwise, it returns an empty list.

        Returns:
            list: A list of dataset subset names. Returns an empty list if the file does not exist.
        """
        cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
        data_cache_dir = cache_manager.get_cache_dir("data_cache")
        model_txt_path = os.path.join(data_cache_dir, "model.txt")
        
        if not os.path.exists(model_txt_path):
            return []
        
        with open(model_txt_path, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]

    @staticmethod
    def build_splits(subset):
        """
        Builds training and test splits for a given dataset subset.

        This function first checks if the dataset subset exists. If it does, 
        it attempts to perform a 90/10 train-test split. If the dataset doesn't 
        support the train_test_split method, it uses a simple selection for the test set.
        Finally, the split dataset is saved back to disk.

        Args:
            subset (str): The name of the dataset subset.

        Returns:
            None: Returns None if the dataset subset does not exist.
        """
        cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
        subset_path = os.path.join(cache_manager.get_cache_dir("data_cache"), subset)
        
        if not os.path.exists(subset_path):
            return
        
        dataset = load_from_disk(subset_path)
        
        if hasattr(dataset, "train_test_split"):
            split_data = dataset.train_test_split(test_size=0.1, seed=42)
        else:
            split_data = {"train": dataset, "test": dataset.select(range(min(1000, len(dataset))))}
        
        dataset_dict = DatasetDict(split_data)
        dataset_dict.save_to_disk(subset_path)

if __name__ == "__main__":
    SUBSETS = PiscesLxToolsDataPreprocessor.get_subsets_from_model_txt()
    for subset in SUBSETS:
        PiscesLxToolsDataPreprocessor.build_splits(subset)