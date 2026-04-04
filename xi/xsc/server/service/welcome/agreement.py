#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Agreement Handler for Welcome Screen

Handles agreement-related operations including:
- Agreement content loading
- First launch completion
- Agreement acceptance tracking
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ....core.dc import XiLogger
from ....config import get_xi_config, get_config_loader


@dataclass
class AgreementInfo:
    """Information about the agreement."""
    content: str
    version: str
    path: str
    exists: bool


class PiscesL1WelcomeAgreement:
    """
    Agreement handler for Xi Studio Welcome screen.
    
    Handles:
    - Loading agreement content from file
    - Completing first launch
    - Agreement acceptance status
    """
    
    DEFAULT_AGREEMENT_VERSION = "1.0.0"
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = XiLogger("Xi.WelcomeAgreement", enable_file=True)
        self._agreement_path: Optional[Path] = None
        self._content: Optional[str] = None
    
    def get_agreement_path(self) -> Path:
        """Get the path to the agreement file."""
        if self._agreement_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
            self._agreement_path = data_dir / "agreement.txt"
        return self._agreement_path
    
    def get_agreement(self) -> AgreementInfo:
        """
        Load and return the agreement content.
        
        Returns:
            AgreementInfo with content and metadata
        """
        agreement_path = self.get_agreement_path()
        
        if not agreement_path.exists():
            self.logger.warning(
                f"Agreement file not found: {agreement_path}",
                event="xi.agreement.not_found"
            )
            return AgreementInfo(
                content="",
                version=self.DEFAULT_AGREEMENT_VERSION,
                path=str(agreement_path),
                exists=False
            )
        
        try:
            with open(agreement_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            self._content = content
            
            version = self._extract_version(content)
            
            self.logger.debug(
                f"Agreement loaded: {len(content)} bytes",
                event="xi.agreement.loaded"
            )
            
            return AgreementInfo(
                content=content,
                version=version,
                path=str(agreement_path),
                exists=True
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to read agreement: {e}",
                event="xi.agreement.read_error"
            )
            return AgreementInfo(
                content="",
                version=self.DEFAULT_AGREEMENT_VERSION,
                path=str(agreement_path),
                exists=False
            )
    
    def _extract_version(self, content: str) -> str:
        """
        Extract version from agreement content.
        
        Looks for version pattern like 'Version: 1.0.0' or 'v1.0.0'.
        
        Args:
            content: Agreement content
            
        Returns:
            Version string or default version
        """
        patterns = [
            r'[Vv]ersion:\s*(\d+\.\d+\.\d+)',
            r'[Vv](\d+\.\d+\.\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return self.DEFAULT_AGREEMENT_VERSION
    
    def complete_first_launch(self) -> bool:
        """
        Mark first launch as complete by updating xi.toml.
        
        Returns:
            True if successful, False otherwise
        """
        xi_dir = self.project_root / ".xi"
        xi_toml_path = xi_dir / "xi.toml"
        
        if not xi_toml_path.exists():
            self.logger.error(
                "xi.toml not found",
                event="xi.agreement.config_not_found"
            )
            return False
        
        try:
            with open(xi_toml_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = re.sub(
                r'first_launch\s*=\s*\w+',
                'first_launch = false',
                content,
                flags=re.IGNORECASE
            )
            
            if 'first_launch' not in content:
                if content.strip() and not content.endswith('\n'):
                    new_content = content + '\n\n[project]\nfirst_launch = false\n'
                else:
                    new_content = content + '[project]\nfirst_launch = false\n'
            
            with open(xi_toml_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
            
            get_config_loader().reload()
            
            self.logger.info(
                "First launch marked as complete",
                event="xi.agreement.first_launch_complete"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to complete first launch: {e}",
                event="xi.agreement.complete_error"
            )
            return False
    
    def is_first_launch(self) -> bool:
        """
        Check if this is the first launch.
        
        Returns:
            True if first launch, False otherwise
        """
        xi_dir = self.project_root / ".xi"
        xi_toml_path = xi_dir / "xi.toml"
        
        if not xi_toml_path.exists():
            return True
        
        try:
            import tomllib
            with open(xi_toml_path, "rb") as f:
                data = tomllib.load(f)
            
            project = data.get("project", {})
            first_launch = project.get("first_launch", True)
            
            if isinstance(first_launch, str):
                return first_launch.lower() not in ("false", "0", "no", "off")
            
            return bool(first_launch)
            
        except Exception as e:
            self.logger.error(
                f"Failed to check first launch: {e}",
                event="xi.agreement.check_error"
            )
            return True


_agreement_instance: Optional[PiscesL1WelcomeAgreement] = None


def get_welcome_agreement(project_root: Optional[Path] = None) -> PiscesL1WelcomeAgreement:
    """Get or create the global welcome agreement instance."""
    global _agreement_instance
    if _agreement_instance is None:
        _agreement_instance = PiscesL1WelcomeAgreement(project_root)
    return _agreement_instance
