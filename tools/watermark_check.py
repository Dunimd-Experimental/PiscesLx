#!/usr/bin/env/python3

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
import json
import argparse
from typing import Dict, Any, Optional
from utils.log import RIGHT, DEBUG, ERROR
from tools.watermark import check_text_watermark

def detect_watermark(text: str, verbose: bool = False) -> Dict[str, Any]:
    """Detect hidden watermark information in the text.

    Args:
        text: The text to be detected.
        verbose: Whether to display detailed information.

    Returns:
        A dictionary containing detection results.
    """
    result = {
        "watermark_detected": False,
        "watermark_info": None,
        "compliance_status": "unknown",
        "error": None
    }
    
    try:
        watermark_info = check_text_watermark(text)
        
        if watermark_info:
            result["watermark_detected"] = True
            result["watermark_info"] = watermark_info
            result["compliance_status"] = "compliant"

            if verbose:
                RIGHTt("Valid watermark detected")
                print(f"\tModel: {watermark_info.get('model', 'unknown')}")
                print(f"\tVersion: {watermark_info.get('version', 'unknown')}")
                print(f"\tGeneration Time: {watermark_info.get('timestamp', 'unknown')}")
                print(f"\tSession ID: {watermark_info.get('session_id', 'unknown')}")
                print(f"\tCompliance Standard: {watermark_info.get('standard', 'unknown')}")

                params = watermark_info.get("generation_params", {})
                if params:
                    print(f"\tGeneration Parameters: {json.dumps(params, ensure_ascii=False)}")
        else:
            result["compliance_status"] = "no_watermark"
            if verbose:
                ERROR("No watermark detected")
                
    except Exception as e:
        result["error"] = str(e)
        result["compliance_status"] = "error"
        if verbose:
            ERROR(f"Detection error: {e}")
    
    return result

def batch_detect(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Batch detect watermarks in a file.

    Args:
        file_path: Path to the text file.
        verbose: Whether to display detailed information.

    Returns:
        A dictionary containing batch detection results.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        results = []

        for i, line in enumerate(lines, 1):
            if line.strip():
                result = detect_watermark(line.strip(), verbose=False)
                result["line_number"] = i
                result["line_content"] = line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                results.append(result)

        total_lines = len(results)
        detected_lines = sum(1 for r in results if r["watermark_detected"])
        compliant_lines = sum(1 for r in results if r["compliance_status"] == "compliant")

        summary = {
            "total_lines": total_lines,
            "detected_lines": detected_lines,
            "compliant_lines": compliant_lines,
            "detection_rate": detected_lines / total_lines if total_lines > 0 else 0,
            "compliance_rate": compliant_lines / total_lines if total_lines > 0 else 0,
            "detailed_results": results
        }

        if verbose:
            RIGHT(f"Batch Detection Results:")
            print(f"\tTotal Lines: {total_lines}")
            print(f"\tLines with Watermark Detected: {detected_lines}")
            print(f"\tCompliant Lines: {compliant_lines}")
            print(f"\tDetection Rate: {summary['detection_rate']:.2%}")
            print(f"\tCompliance Rate: {summary['compliance_rate']:.2%}")

        return summary
        
    except Exception as e:
        return {
            "error": str(e),
            "compliance_status": "error"
        }