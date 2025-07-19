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

import os, json
from datasets import load_from_disk, DatasetDict


def get_subsets_from_model_txt():
    model_txt = os.path.join("data_cache", "model.txt")
    if not os.path.exists(model_txt):
        print(f"❌\t{model_txt} not found! Please create it with one dataset name per line.")
        return []
    with open(model_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def build_splits(subset):
    src = f"data/{subset}"
    if not os.path.exists(src):
        print(f"❌\t{src} does not exist, please run download.py first")
        return

    ds = load_from_disk(src)

    # Simple 90/10 split
    if hasattr(ds, "train_test_split"):
        split = ds.train_test_split(test_size=0.1, seed=42)
    else:
        split = {"train": ds, "test": ds.select(range(min(1000, len(ds))))}

    out = DatasetDict(split)
    out.save_to_disk(f"data/{subset}")
    print(f"✅\t{subset} split completed → data/{subset}")

if __name__ == "__main__":
    SUBSETS = get_subsets_from_model_txt()
    for s in SUBSETS:
        build_splits(s)