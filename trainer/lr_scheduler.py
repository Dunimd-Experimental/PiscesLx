#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of LingSi.
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

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Create and return a CosineAnnealingWarmRestarts scheduler instance.
#
# Args:
#     optimizer (torch.optim.Optimizer): Wrapped optimizer.
#     T_0 (int, optional): Number of iterations for the first restart. Defaults to 1000.
#
# Returns:
#     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts: A CosineAnnealingWarmRestarts scheduler instance.
def get_scheduler(optimizer, T_0=1000):
    return CosineAnnealingWarmRestarts(optimizer, T_0)