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

import torch, os
from utils.log import RIGHT
from model import PiscesModel, PiscesConfig
from transformers import BitsAndBytesConfig

def quantize(checkpoint, save_path, bits=8):
    cfg = PiscesConfig.from_json("configs/0.5B.json")
    model = PiscesModel(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    if bits == 8:
        import bitsandbytes as bnb
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight = bnb.nn.Params8bit(m.weight)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    RIGHT("Quantized model saved to", save_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()
    quantize(args.ckpt, args.save, args.bits)