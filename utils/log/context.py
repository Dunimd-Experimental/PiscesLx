#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

import contextvars
from typing import Dict, Any

class PiscesLxCoreLogContext:
    """Thread-local logging context (similar to Mapped Diagnostic Context - MDC).

    This class stores contextual keys (e.g., request_id, session_id, component) 
    that will be merged into every log record.
    """

    # Context variable to store the logging context. The default value is an empty dictionary.
    _ctx: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
        "pisces_log_ctx", default={}
    )

    @classmethod
    def put(cls, **kv: Any) -> None:
        """Update the logging context with the given key-value pairs.

        Retrieves the current logging context, updates it with the provided key-value pairs,
        and then sets the updated context back.

        Args:
            **kv (Any): Key-value pairs to update the logging context with.
        """
        d = dict(cls._ctx.get())
        d.update(kv)
        cls._ctx.set(d)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Retrieve a copy of the current logging context.

        Returns:
            Dict[str, Any]: A copy of the current logging context dictionary.
        """
        return dict(cls._ctx.get())

    @classmethod
    def clear(cls) -> None:
        """Clear the logging context.

        Sets the logging context to an empty dictionary.
        """
        cls._ctx.set({})
