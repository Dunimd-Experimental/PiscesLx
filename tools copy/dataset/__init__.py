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
import sys
import subprocess
from pathlib import Path

def dataset(args=None):
    """
    Used to launch the dataset page via the command line from external entry points such as manage.py.
    Forces execution via the command line and uses absolute paths to avoid issues caused by the current working directory (CWD).
    """
    # Resolve the absolute path to the main.py file
    main_path = Path(__file__).resolve().parent / 'main.py'
    # Run the streamlit application using the resolved path
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', str(main_path)
    ])