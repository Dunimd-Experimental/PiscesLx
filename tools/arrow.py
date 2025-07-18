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

import os
import json
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

def arrow(args):
    if args.json_dir and args.arrow_out:
        json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith('.json')]
        if not json_files:
            print(f"❌\tNo .json files found in {args.json_dir}")
            return
        all_data = []
        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line))
                    except Exception as e:
                        print(f"❌\tError parsing {jf}: {e}")
        if not all_data:
            print("❌\tNo data loaded from json files.")
            return
        ds = Dataset.from_list(all_data)
        ds.save_to_disk(args.arrow_out)
        print(f"✅\tSaved {len(ds)} samples to {args.arrow_out}")
        return
    elif args.arrow_in and args.json_out:
        if not os.path.exists(args.arrow_in):
            print(f"❌\tArrow file not found: {args.arrow_in}")
            return
        ds = load_from_disk(args.arrow_in)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅\tSaved {len(ds)} samples to {args.json_out}")
        return
    else:
        print("❌\tPlease specify either --json_dir + --arrow_out or --arrow_in + --json_out")
        print("For example：")
        print("\tpython manage.py arrow --json_dir ./jsons --arrow_out ./out.arrow")
        print("\tpython manage.py arrow --arrow_in ./in.arrow --json_out ./out.json") 