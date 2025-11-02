#!/usr/bin/env python3

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

import sys
import json
import argparse
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.data.download")
from typing import Dict, Any, Optional
from tools.infer.watermark import check_text_watermark

def detect_watermark(text: str, verbose: bool = False) -> Dict[str, Any]:
    """Detect hidden watermark information in the specified text.

    Args:
        text (str): The text to be detected for watermarks.
        verbose (bool, optional): Whether to display detailed detection information. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing detection results with the following keys:
            - watermark_detected (bool): Indicates whether a watermark was detected.
            - watermark_info (Dict[str, Any]): Information about the detected watermark, None if no watermark is detected.
            - compliance_status (str): Compliance status, possible values are "unknown", "compliant", "no_watermark", or "error".
            - error (str): Error message if an exception occurs during detection, None otherwise.
    """
    # Validate input parameters
    _validate_detect_args(text, verbose)

    # Initialize the result dictionary
    result = {
        "watermark_detected": False,
        "watermark_info": None,
        "compliance_status": "unknown",
        "error": None
    }
    
    try:
        # Check for watermark in the text
        watermark_info = check_text_watermark(text)
        
        if watermark_info:
            # Update result if a watermark is detected
            result["watermark_detected"] = True
            result["watermark_info"] = watermark_info
            result["compliance_status"] = "compliant"

            if verbose:
                logger.info("Valid watermark detected")
                print(f"\tModel: {watermark_info.get('model', 'unknown')}")
                print(f"\tVersion: {watermark_info.get('version', 'unknown')}")
                print(f"\tGeneration Time: {watermark_info.get('timestamp', 'unknown')}")
                print(f"\tSession ID: {watermark_info.get('session_id', 'unknown')}")
                print(f"\tCompliance Standard: {watermark_info.get('standard', 'unknown')}")

                params = watermark_info.get("generation_params", {})
                if params:
                    print(f"\tGeneration Parameters: {json.dumps(params, ensure_ascii=False)}")
        else:
            # Update result if no watermark is detected
            result["compliance_status"] = "no_watermark"
            if verbose:
                logger.error("No watermark detected")
                
    except Exception as e:
        # Update result if an error occurs during detection
        result["error"] = str(e)
        result["compliance_status"] = "error"
        if verbose:
            logger.error(f"Detection error: {e}")
    
    return result

def batch_detect(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Batch detect watermarks in a text file.

    Args:
        file_path (str): Path to the text file for batch watermark detection.
        verbose (bool, optional): Whether to display detailed batch detection results. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing batch detection results with the following keys:
            - total_lines (int): Total number of lines processed in the file.
            - detected_lines (int): Number of lines with watermarks detected.
            - compliant_lines (int): Number of compliant lines.
            - detection_rate (float): Watermark detection rate.
            - compliance_rate (float): Compliance rate.
            - detailed_results (List[Dict[str, Any]]): Detailed detection results for each line.
            - error (str): Error message if an exception occurs during batch detection, None otherwise.
            - compliance_status (str): Compliance status, "error" if an exception occurs.
    """
    # Validate input parameters
    _validate_batch_detect_args(file_path, verbose)
    try:
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content into lines
        lines = content.split('\n')
        results = []

        # Detect watermarks in each non-empty line
        for i, line in enumerate(lines, 1):
            if line.strip():
                result = detect_watermark(line.strip(), verbose=False)
                result["line_number"] = i
                result["line_content"] = line.strip()[:100] + "..." if len(line) > 100 else line.strip()
                results.append(result)

        # Calculate summary statistics
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
            logger.info(f"Batch Detection Results:")
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

def _validate_detect_args(text: str, verbose: bool):
    """Validate the input parameters for the detect_watermark function.

    Args:
        text (str): The text to be validated.
        verbose (bool): The verbose flag to be validated.

    Raises:
        ValueError: If text is not a non-empty string or verbose is not a boolean value.
    """
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("text must be a non-empty string")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")

def _validate_batch_detect_args(file_path: str, verbose: bool):
    """Validate the input parameters for the batch_detect function.

    Args:
        file_path (str): The file path to be validated.
        verbose (bool): The verbose flag to be validated.

    Raises:
        ValueError: If file_path is not a non-empty string, the file does not exist, 
                    or verbose is not a boolean value.
    """
    if not isinstance(file_path, str) or file_path.strip() == "":
        raise ValueError("file_path must be a non-empty string")
    import os
    if not os.path.exists(file_path):
        raise ValueError(f"file_path not found: {file_path}")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")