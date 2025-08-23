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
from utils.log import RIGHT, DEBUG, ERROR

def update():
    """
    Pull latest code from remote repository.
    """
    remote_url = 'https://gitee.com/dunimd/piscesl1.git'
    try:
        DEBUG(f"Pulling latest code from {remote_url}...")
        subprocess.run(['git', 'fetch', '--all'], check=True)
        subprocess.run(['git', 'reset', '--hard', 'origin/master'], check=True)
        RIGHT("Code successfully updated to the latest version")
    except subprocess.CalledProcessError as e:
        ERROR(f"Failed to pull code: {e}")
        sys.exit(1)