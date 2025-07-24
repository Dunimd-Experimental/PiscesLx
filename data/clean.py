#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import re
import os
from datasets import load_from_disk, Dataset

class DatasetCleaner:
    @staticmethod
    def clean_text(text, min_length=10, keep_pattern=r'[\u4e00-\u9fff，。！？、：“”‘’\d\w\s]'):
        if not text or not isinstance(text, str):
            return None
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        if keep_pattern:
            text = re.sub(f'[^{keep_pattern}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) >= min_length else None

    @staticmethod
    def process_dataset(input_path, output_path, text_field='text', **clean_kwargs):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dataset = load_from_disk(input_path)
        cleaned_data = []
        for sample in dataset:
            text = sample.get(text_field, '')
            cleaned_text = DatasetCleaner.clean_text(text, **clean_kwargs)
            if cleaned_text:
                new_sample = {k: v for k, v in sample.items() if k != text_field}
                new_sample[text_field] = cleaned_text
                cleaned_data.append(new_sample)

        cleaned_dataset = Dataset.from_list(cleaned_data)
        cleaned_dataset.save_to_disk(output_path)
        return len(cleaned_data), len(dataset)

    @staticmethod
    def auto_clean(input_dir, output_dir='data_clean', **clean_kwargs):
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        for dataset_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, dataset_name)
            if os.path.isdir(input_path) and not dataset_name.endswith('_clean'):
                output_path = os.path.join(output_dir, f"{dataset_name}_clean")
                if not os.path.exists(output_path):
                    cleaned_count, total_count = DatasetCleaner.process_dataset(
                        input_path, output_path, **clean_kwargs
                    )
                    print(f"✅\tCleaning completed: {dataset_name} -> {dataset_name}_clean | Samples retained: {cleaned_count}/{total_count}")
                else:
                    print(f"✅\tCleaned dataset already exists: {output_path}, skipping processing")
        return True