#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import re

def natural_sort_key(text):
    """
    Generate a key for natural sorting, which can handle numeric parts in the text.
    Natural sorting is a way to sort strings containing numbers in a human-friendly order,
    for example, it will sort ["file1", "file10", "file2"] as ["file1", "file2", "file10"].

    Args:
        text (str): The input text to generate the sorting key for.

    Returns:
        list: A list containing integers or lowercase strings, used as the sorting key.
    """
    # Split the text by digits, then convert digit parts to integers and non-digit parts to lowercase
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]
