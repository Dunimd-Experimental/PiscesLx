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

import time, torch
from model import PiscesModel, PiscesConfig


cfg = PiscesConfig.from_json("configs/0.5B.json")
# Smart device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅\tUsing device: {device}")

model = PiscesModel(cfg).to(device).eval()
tok = torch.randint(0, cfg.vocab_size, (1, 8192)).to(device)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    _ = model(tok)
torch.cuda.synchronize()

print("\t8192 tokens forward:", time.time()-t0, "s")