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

def help():
    print("Pisces L1 Management Tool Help\n")
    print("Available commands:")
    print("  setup      - Environment setup and dependency installation")
    print("  train      - Train the model")
    print("  infer      - Run inference with a trained model")
    print("  check      - Check GPU and dependencies")
    print("  monitor    - System monitor (GPU/CPU/memory)")
    print("  download   - Download datasets for training")
    print("  arrow      - Arrow/JSON dataset conversion")
    print("  help       - Show this help message")
    print("\nExample usage:")
    print("  python manage.py train")
    print("  python manage.py infer --ckpt ckpt/model.pt --prompt 'Hello'")
    print("  python manage.py help\n") 