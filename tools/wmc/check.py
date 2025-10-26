#!/usr/bin/env python3

# Watermark checker utilities (migrated from tools/watermark_check.py)

import os
import json
from typing import Dict, Any, Optional

from utils import PiscesLxCoreLog
from utils.watermark.content import PiscesWatermark

logger = PiscesLxCoreLog("pisceslx.tools.wmc")


def _validate_detect_args(text: str, verbose: bool):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("text must be a non-empty string")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")


def _validate_batch_detect_args(file_path: str, verbose: bool):
    if not isinstance(file_path, str) or file_path.strip() == "":
        raise ValueError("file_path must be a non-empty string")
    if not os.path.exists(file_path):
        raise ValueError(f"file_path not found: {file_path}")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")


def detect_watermark(text: str, verbose: bool = False) -> Dict[str, Any]:
    """Detect hidden watermark information in the specified text.

    Returns a dict: {
      watermark_detected: bool,
      watermark_info: Optional[dict],
      compliance_status: "unknown"|"compliant"|"no_watermark"|"error",
      error: Optional[str]
    }
    """
    _validate_detect_args(text, verbose)

    result: Dict[str, Any] = {
        "watermark_detected": False,
        "watermark_info": None,
        "compliance_status": "unknown",
        "error": None,
    }

    try:
        wm = PiscesWatermark()
        payload = wm.extract_text_watermark(text)
        if payload:
            result["watermark_detected"] = True
            result["watermark_info"] = payload
            # naive compliance tagging — refine when protocol is refactored
            std = payload.get("compliance", {}).get("standard") or payload.get("standard")
            result["compliance_status"] = "compliant" if std else "unknown"
            if verbose:
                logger.success("Watermark detected", event="wmc.detect", standard=std or "unknown")
        else:
            result["compliance_status"] = "no_watermark"
            if verbose:
                logger.error("No watermark detected")
    except Exception as e:
        result["error"] = str(e)
        result["compliance_status"] = "error"
        if verbose:
            logger.error(f"Detection error: {e}")

    return result


def batch_detect(file_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Batch detect watermarks line-by-line in a text file."""
    _validate_batch_detect_args(file_path, verbose)

    detected_lines = 0
    compliant_lines = 0
    total_lines = 0
    results = []

    wm = PiscesWatermark()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            text = line.strip()
            if not text:
                continue
            payload = wm.extract_text_watermark(text)
            detected = payload is not None
            compliant = False
            std = None
            if detected:
                detected_lines += 1
                std = payload.get("compliance", {}).get("standard") or payload.get("standard")
                compliant = bool(std)
                if compliant:
                    compliant_lines += 1
            results.append({
                "line": total_lines,
                "detected": detected,
                "compliant": compliant,
                "standard": std,
            })

    summary = {
        "total_lines": total_lines,
        "detected_lines": detected_lines,
        "compliant_lines": compliant_lines,
        "detection_rate": (detected_lines / total_lines) if total_lines else 0.0,
        "compliance_rate": (compliant_lines / total_lines) if total_lines else 0.0,
        "detailed_results": results,
    }

    if verbose:
        logger.info("Batch Detection Results")
        print(f"\tTotal Lines: {total_lines}")
        print(f"\tLines with Watermark Detected: {detected_lines}")
        print(f"\tCompliant Lines: {compliant_lines}")
        print(f"\tDetection Rate: {summary['detection_rate']:.2%}")
        print(f"\tCompliance Rate: {summary['compliance_rate']:.2%}")

    return summary


# -------- Image / Audio detection (file-based) --------

def _load_image_as_tensor(path: str):
    try:
        from PIL import Image
        import torch
        img = Image.open(path).convert('RGB')
        t = torch.from_numpy(__import__('numpy').array(img)).permute(2, 0, 1).float() / 255.0
        return t
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def _load_audio_as_tensor(path: str):
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav, int(sr)
    except Exception:
        try:
            import wave, struct, torch
            with wave.open(path, 'rb') as wf:
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                raw = wf.readframes(n_frames)
                fmt = {1:'b',2:'h',4:'i'}.get(sampwidth)
                if fmt is None:
                    raise RuntimeError("Unsupported sample width")
                import array
                arr = array.array(fmt, raw)
                import numpy as np
                np_arr = np.asarray(arr, dtype=float)
                if n_channels > 1:
                    np_arr = np_arr.reshape(-1, n_channels).T
                else:
                    np_arr = np_arr.reshape(1, -1)
                wav = torch.from_numpy(np_arr).float() / (2**(8*sampwidth-1))
                return wav, int(framerate)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, None


def detect_image_watermark(image_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Detect watermark payload from an image file."""
    img_t = _load_image_as_tensor(image_path)
    result = {"file": image_path, "watermark_detected": False, "watermark_info": None, "error": None}
    if img_t is None:
        result["error"] = "image_load_failed"
        return result
    try:
        wm = PiscesWatermark()
        payload = wm.extract_image_watermark(img_t)
        if payload:
            result["watermark_detected"] = True
            result["watermark_info"] = payload
            if verbose:
                logger.success("Image watermark detected", event="wmc.img.detect")
        else:
            if verbose:
                logger.error("No image watermark detected")
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def detect_audio_watermark(audio_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Detect watermark payload from an audio file."""
    wav, sr = _load_audio_as_tensor(audio_path)
    result = {"file": audio_path, "watermark_detected": False, "watermark_info": None, "error": None}
    if wav is None or sr is None:
        result["error"] = "audio_load_failed"
        return result
    try:
        wm = PiscesWatermark()
        payload = wm.extract_audio_watermark(wav, sample_rate=sr)
        if payload:
            result["watermark_detected"] = True
            result["watermark_info"] = payload
            if verbose:
                logger.success("Audio watermark detected", event="wmc.audio.detect")
        else:
            if verbose:
                logger.error("No audio watermark detected")
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
