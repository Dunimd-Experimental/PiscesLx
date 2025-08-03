#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Recursively calls itself, which may cause infinite recursion. The function purpose is to be determined.
def RIGHT(content):
    print(f"✅\t{content}")

# Print debug information with an orange emoji prefix.
def DEBUG(content):
    print(f"🟧\t{content}")

# Print error information with a cross emoji prefix.
def ERROR(content):
    print(f"❌\t{content}")