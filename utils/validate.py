#!/usr/bin/env/python3

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

from typing import Any, Iterable, Tuple, Type
from utils.error import PiscesLxCoreValidationError

class PiscesLxCoreValidator:
    """A lightweight helper class for validation and assertions.
    
    Raises PiscesLxCoreValidationError with consistent error messages.
    """

    @staticmethod
    def require(condition: bool, message: str, **ctx: Any) -> None:
        """Raise a validation error if the given condition is not met.

        Args:
            condition (bool): The condition to check.
            message (str): The error message to raise if the condition is False.
            **ctx: Additional context information to include in the error.

        Raises:
            PiscesLxCoreValidationError: If the `condition` is False.
        """
        if not condition:
            raise PiscesLxCoreValidationError(message, context=ctx or None)

    @staticmethod
    def expect_type(obj: Any, types: Tuple[Type, ...] | Type, name: str) -> None:
        """Verify that an object is of the expected type(s).

        Args:
            obj (Any): The object to check.
            types (Tuple[Type, ...] | Type): The expected type or tuple of types.
            name (str): The name of the object for error reporting.

        Raises:
            PiscesLxCoreValidationError: If `obj` is not an instance of `types`.
        """
        if not isinstance(obj, types):
            # Get the name of the expected type(s)
            tname = getattr(types, "__name__", str(types)) if isinstance(types, type) else \
                "|".join(getattr(t, "__name__", str(t)) for t in types)  # type: ignore[arg-type]
            raise PiscesLxCoreValidationError(
                "invalid type",
                context={"name": name, "expected": tname, "got": type(obj).__name__},
            )

    @staticmethod
    def expect_in(value: Any, options: Iterable[Any], name: str) -> None:
        """Verify that a value is among the expected options.

        Args:
            value (Any): The value to check.
            options (Iterable[Any]): An iterable containing the expected options.
            name (str): The name of the value for error reporting.

        Raises:
            PiscesLxCoreValidationError: If `value` is not in `options`.
        """
        if value not in options:
            raise PiscesLxCoreValidationError(
                "invalid value",
                context={"name": name, "value": value, "options": list(options)},
            )
