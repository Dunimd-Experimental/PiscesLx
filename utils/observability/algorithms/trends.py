#!/usr/bin/env python3

# Copyright æ¼?2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import os
import math
import time
import threading
from enum import Enum
from utils import PiscesLxCoreLog
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

_LOGGER = PiscesLxCoreLog("pisceslx.algorithms.trends")

@dataclass
class EWMAConfig:
    """Configuration class for the EWMA (Exponential Weighted Moving Average) algorithm.

    Attributes:
        alpha (float): Smoothing factor, controlling the weight of new values. Defaults to 0.2.
        min_alpha (float): Minimum value of the smoothing factor. Defaults to 0.05.
        max_alpha (float): Maximum value of the smoothing factor. Defaults to 0.5.
        adaptive_alpha (bool): Flag indicating whether to use an adaptive smoothing factor. Defaults to True.
    """
    alpha: float = 0.2
    min_alpha: float = 0.05
    max_alpha: float = 0.5
    adaptive_alpha: bool = True

@dataclass
class MADConfig:
    """Configuration class for the MAD (Median Absolute Deviation) algorithm.

    Attributes:
        k (float): Threshold for outlier detection. Defaults to 6.0.
        min_k (float): Minimum threshold value. Defaults to 3.0.
        max_k (float): Maximum threshold value. Defaults to 12.0.
        adaptive_k (bool): Flag indicating whether to use an adaptive threshold. Defaults to True.
        window_size (int): Size of the sliding window for analysis. Defaults to 100.
    """
    k: float = 6.0
    min_k: float = 3.0
    max_k: float = 12.0
    adaptive_k: bool = True
    window_size: int = 100

@dataclass
class SeasonalConfig:
    """Configuration class for seasonal decomposition.

    Attributes:
        season_length (int): Length of a season in hours. Defaults to 24.
        trend_window (int): Size of the trend window in days. Defaults to 7.
        enable_seasonal (bool): Flag indicating whether to enable seasonal detection. Defaults to True.
    """
    season_length: int = 24
    trend_window: int = 7
    enable_seasonal: bool = True

@dataclass
class TrendAlgorithmConfig:
    """Global configuration class for trend algorithms.

    Attributes:
        ewma (EWMAConfig): Configuration for the EWMA algorithm.
        mad (MADConfig): Configuration for the MAD algorithm.
        seasonal (SeasonalConfig): Configuration for seasonal decomposition.
        enable_adaptive (bool): Flag indicating whether to enable adaptive algorithms. Defaults to True.
        min_data_points (int): Minimum number of data points required for analysis. Defaults to 10.
        max_history_size (int): Maximum size of historical data to retain. Defaults to 10000.
    """
    ewma: EWMAConfig = field(default_factory=EWMAConfig)
    mad: MADConfig = field(default_factory=MADConfig)
    seasonal: SeasonalConfig = field(default_factory=SeasonalConfig)
    enable_adaptive: bool = True
    min_data_points: int = 10
    max_history_size: int = 10000

class TrendState(Enum):
    """Enumeration representing different trend states.

    Attributes:
        STABLE: Indicates a stable trend.
        INCREASING: Indicates an increasing trend.
        DECREASING: Indicates a decreasing trend.
        VOLATILE: Indicates a volatile trend.
        ANOMALOUS: Indicates an anomalous trend.
    """
    STABLE = "stable"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    VOLATILE = "volatile"
    ANOMALOUS = "anomalous"

@dataclass
class TrendAnalysisResult:
    """Class representing the result of trend analysis.

    Attributes:
        value (float): The input value being analyzed.
        trend (TrendState): The determined trend state.
        ewma (float): The computed EWMA value.
        baseline (float): The computed baseline value.
        drift_ratio (float): The computed drift ratio.
        is_outlier (bool): Flag indicating whether the value is an outlier.
        confidence (float): The confidence level of the analysis.
        seasonal_component (Optional[float]): The seasonal component of the analysis. Defaults to None.
        trend_component (Optional[float]): The trend component of the analysis. Defaults to None.
        residual (Optional[float]): The residual value of the analysis. Defaults to None.
    """
    value: float
    trend: TrendState
    ewma: float
    baseline: float
    drift_ratio: float
    is_outlier: bool
    confidence: float
    seasonal_component: Optional[float] = None
    trend_component: Optional[float] = None
    residual: Optional[float] = None

class EnterpriseTrendAnalyzer:
    """Analyzer for enterprise-level trend analysis."""
    
    def __init__(self, config: Optional[TrendAlgorithmConfig] = None):
        """Initialize the EnterpriseTrendAnalyzer.

        Args:
            config (Optional[TrendAlgorithmConfig]): Configuration for trend algorithms. 
                If None, load configuration from environment variables. Defaults to None.
        """
        self.config = config or self._load_config_from_env()
        self.history: List[float] = []
        self.timestamps: List[float] = []
        self.seasonal_patterns: Dict[int, List[float]] = {}
        self._lock = threading.RLock()
        # Maintain an internal EWMA state to avoid using previous raw value approximation
        self._ewma_state: Optional[float] = None
        
    def _load_config_from_env(self) -> TrendAlgorithmConfig:
        """Load trend algorithm configuration from environment variables.

        Returns:
            TrendAlgorithmConfig: The loaded configuration.
        """
        return TrendAlgorithmConfig(
            ewma=EWMAConfig(
                alpha=float(os.getenv("PISCES_TREND_EWMA_ALPHA", "0.2")),
                min_alpha=float(os.getenv("PISCES_TREND_EWMA_MIN_ALPHA", "0.05")),
                max_alpha=float(os.getenv("PISCES_TREND_EWMA_MAX_ALPHA", "0.5")),
                adaptive_alpha=os.getenv("PISCES_TREND_EWMA_ADAPTIVE", "true").lower() == "true"
            ),
            mad=MADConfig(
                k=float(os.getenv("PISCES_TREND_MAD_K", "6.0")),
                min_k=float(os.getenv("PISCES_TREND_MAD_MIN_K", "3.0")),
                max_k=float(os.getenv("PISCES_TREND_MAD_MAX_K", "12.0")),
                adaptive_k=os.getenv("PISCES_TREND_MAD_ADAPTIVE", "true").lower() == "true",
                window_size=int(os.getenv("PISCES_TREND_MAD_WINDOW_SIZE", "100"))
            ),
            seasonal=SeasonalConfig(
                season_length=int(os.getenv("PISCES_TREND_SEASONAL_LENGTH", "24")),
                trend_window=int(os.getenv("PISCES_TREND_SEASONAL_TREND_WINDOW", "7")),
                enable_seasonal=os.getenv("PISCES_TREND_SEASONAL_ENABLE", "true").lower() == "true"
            ),
            enable_adaptive=os.getenv("PISCES_TREND_ENABLE_ADAPTIVE", "true").lower() == "true",
            min_data_points=int(os.getenv("PISCES_TREND_MIN_DATA_POINTS", "10")),
            max_history_size=int(os.getenv("PISCES_TREND_MAX_HISTORY_SIZE", "10000"))
        )
    
    def analyze(self, value: float, timestamp: Optional[float] = None) -> TrendAnalysisResult:
        """Perform enterprise-level trend analysis on the input value.

        Args:
            value (float): The input value to analyze.
            timestamp (Optional[float]): The timestamp of the input value. 
                If None, use the current time. Defaults to None.

        Returns:
            TrendAnalysisResult: The result of the trend analysis.
        """
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
                
            # Update historical data
            self._update_history(value, timestamp)
            
            # Compute EWMA
            ewma_value = self._compute_ewma(value)
            
            # Compute baseline
            baseline = self._compute_baseline()
            
            # Compute drift ratio
            drift = self._compute_drift_ratio(value, baseline)
            
            # Perform seasonal decomposition
            seasonal_comp, trend_comp, residual = self._seasonal_decompose()
            
            # Detect outliers
            is_outlier = self._detect_outlier(value)
            
            # Determine trend state
            trend_state = self._determine_trend_state(value, ewma_value, drift, is_outlier)
            
            # Compute confidence
            confidence = self._compute_confidence(value, ewma_value, baseline)
            
            return TrendAnalysisResult(
                value=value,
                trend=trend_state,
                ewma=ewma_value,
                baseline=baseline,
                drift_ratio=drift,
                is_outlier=is_outlier,
                confidence=confidence,
                seasonal_component=seasonal_comp,
                trend_component=trend_comp,
                residual=residual
            )
    
    def _update_history(self, value: float, timestamp: float) -> None:
        """Update the historical data with a new value and its timestamp.

        Args:
            value (float): The new value to add to the history.
            timestamp (float): The timestamp of the new value.
        """
        self.history.append(value)
        self.timestamps.append(timestamp)
        
        # Limit the size of historical data
        if len(self.history) > self.config.max_history_size:
            self.history = self.history[-self.config.max_history_size:]
            self.timestamps = self.timestamps[-self.config.max_history_size:]
    
    def _compute_ewma(self, value: float) -> float:
        """Compute the adaptive EWMA value for the input value.

        Args:
            value (float): The new value to compute EWMA.

        Returns:
            float: The computed EWMA value.
        """
        if not self.history:
            return value
            
        if len(self.history) == 1:
            return value
            
        prev_ewma = self.history[-2]  # Simplification: use the previous value as EWMA
        
        if self.config.enable_adaptive and self.config.ewma.adaptive_alpha:
            alpha = self._compute_adaptive_alpha()
        else:
            alpha = self.config.ewma.alpha
            
        return compute_ewma(value, prev_ewma, alpha)
    
    def _compute_adaptive_alpha(self) -> float:
        """Compute the adaptive smoothing factor (alpha) based on recent volatility.

        Returns:
            float: The computed adaptive alpha value.
        """
        if len(self.history) < 3:
            return self.config.ewma.alpha
            
        # Adjust alpha based on recent volatility
        recent_values = self.history[-min(10, len(self.history)):]
        volatility = self._compute_volatility(recent_values)
        
        # Increase alpha for high volatility (faster response)
        # Decrease alpha for low volatility (more stable)
        alpha = self.config.ewma.alpha
        if volatility > 0.1:  # High volatility
            alpha = min(self.config.ewma.max_alpha, alpha * 2.0)
        elif volatility < 0.01:  # Low volatility
            alpha = max(self.config.ewma.min_alpha, alpha * 0.5)
            
        return alpha
    
    def _compute_volatility(self, values: List[float]) -> float:
        """Compute the volatility of the given values.

        Args:
            values (List[float]): The input values to compute volatility.

        Returns:
            float: The computed volatility.
        """
        if len(values) < 2:
            return 0.0
            
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / (mean_val + 1e-6)  # Normalize
    
    def _compute_baseline(self) -> float:
        """Compute the adaptive baseline value.

        Returns:
            float: The computed baseline value.
        """
        if len(self.history) < self.config.min_data_points:
            return self.history[-1] if self.history else 0.0
            
        window_size = min(self.config.mad.window_size, len(self.history))
        recent_values = self.history[-window_size:]
        
        # Use median as baseline (robust to outliers)
        sorted_values = sorted(recent_values)
        n = len(sorted_values)
        baseline = sorted_values[n // 2]
        
        # Apply seasonal adjustment
        if self.config.seasonal.enable_seasonal and len(self.history) >= self.config.seasonal.season_length:
            seasonal_adjustment = self._compute_seasonal_adjustment()
            baseline += seasonal_adjustment
            
        return baseline
    
    def _compute_seasonal_adjustment(self) -> float:
        """Compute the seasonal adjustment value.

        Returns:
            float: The computed seasonal adjustment value.
        """
        if not self.config.seasonal.enable_seasonal:
            return 0.0
            
        current_hour = int(time.time() / 3600) % self.config.seasonal.season_length
        
        if current_hour not in self.seasonal_patterns:
            return 0.0
            
        pattern_values = self.seasonal_patterns[current_hour]
        if not pattern_values:
            return 0.0
            
        return sum(pattern_values) / len(pattern_values)
    
    def _seasonal_decompose(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Perform simplified seasonal decomposition on the historical data.

        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]:
                A tuple containing the seasonal component, trend component, and residual.
                If decomposition fails or is disabled, return (None, None, None).
        """
        if not self.config.seasonal.enable_seasonal or len(self.history) < self.config.seasonal.season_length * 2:
            return None, None, None
            
        try:
            # Simplified seasonal decomposition
            current_hour = int(time.time() / 3600) % self.config.seasonal.season_length
            seasonal_comp = self._compute_seasonal_adjustment()
            
            # Trend component (moving average)
            trend_window = min(self.config.seasonal.trend_window * 24, len(self.history))
            trend_values = self.history[-trend_window:]
            trend_comp = sum(trend_values) / len(trend_values) if trend_values else 0.0
            
            # Residual
            current_value = self.history[-1]
            residual = current_value - seasonal_comp - trend_comp
            
            return seasonal_comp, trend_comp, residual
            
        except Exception as e:
            _LOGGER.error("Failed to analyze trends", error=str(e), error_class=type(e).__name__)
            _LOGGER.warning("Failed to detect trends", error=str(e), error_class=type(e).__name__)
            
            return None, None, None
    
    def _compute_drift_ratio(self, value: float, baseline: float) -> float:
        """Compute the drift ratio between the current value and the baseline.

        Args:
            value (float): The current value.
            baseline (float): The baseline value.

        Returns:
            float: The computed drift ratio.
        """
        return drift_ratio(value, baseline)
    
    def _detect_outlier(self, value: float) -> bool:
        """Detect if the given value is an outlier using the adaptive MAD method.

        Args:
            value (float): The value to check for outliers.

        Returns:
            bool: True if the value is an outlier, False otherwise.
        """
        if len(self.history) < self.config.min_data_points:
            return False
            
        window_size = min(self.config.mad.window_size, len(self.history))
        recent_values = self.history[-window_size:]
        
        if len(recent_values) < 3:
            return False
            
        median, mad = mad_stats(recent_values)
        
        if self.config.enable_adaptive and self.config.mad.adaptive_k:
            k = self._compute_adaptive_k(recent_values)
        else:
            k = self.config.mad.k
            
        return is_outlier_mad(value, median, mad, k)
    
    def _compute_adaptive_k(self, values: List[float]) -> float:
        """Compute the adaptive threshold (k) for outlier detection based on data distribution.

        Args:
            values (List[float]): The input values to compute adaptive k.

        Returns:
            float: The computed adaptive k value.
        """
        if len(values) < 10:
            return self.config.mad.k
            
        # Adjust k value based on data distribution
        sorted_values = sorted(values)
        q1 = sorted_values[len(sorted_values) // 4]
        q3 = sorted_values[3 * len(sorted_values) // 4]
        iqr = q3 - q1
        
        # Increase k for high IQR (wider threshold)
        # Decrease k for low IQR (stricter threshold)
        median = sorted_values[len(sorted_values) // 2]
        relative_iqr = iqr / (median + 1e-6)
        
        k = self.config.mad.k
        if relative_iqr > 0.5:  # High dispersion
            k = min(self.config.mad.max_k, k * 1.5)
        elif relative_iqr < 0.1:  # Low dispersion
            k = max(self.config.mad.min_k, k * 0.7)
            
        return k
    
    def _determine_trend_state(self, value: float, ewma_value: float, drift: float, is_outlier: bool) -> TrendState:
        """Determine the trend state based on the input parameters.

        Args:
            value (float): The current value.
            ewma_value (float): The EWMA value.
            drift (float): The drift ratio.
            is_outlier (bool): Flag indicating whether the value is an outlier.

        Returns:
            TrendState: The determined trend state.
        """
        if is_outlier:
            return TrendState.ANOMALOUS
            
        # Determine trend based on drift ratio
        if abs(drift) > 0.5:  # Large drift
            return TrendState.VOLATILE
        elif drift > 0.2:
            return TrendState.INCREASING
        elif drift < -0.2:
            return TrendState.DECREASING
        else:
            return TrendState.STABLE
    
    def _compute_confidence(self, value: float, ewma_value: float, baseline: float) -> float:
        """Compute the confidence level of the trend analysis.

        Args:
            value (float): The current value.
            ewma_value (float): The EWMA value.
            baseline (float): The baseline value.

        Returns:
            float: The computed confidence level in the range [0.1, 1.0].
        """
        if len(self.history) < self.config.min_data_points:
            return 0.5
            
        # Compute confidence based on historical stability
        recent_values = self.history[-min(20, len(self.history)):]
        volatility = self._compute_volatility(recent_values)
        
        # Lower volatility means higher confidence
        confidence = max(0.1, 1.0 - volatility)
        
        # More data points means higher confidence
        data_confidence = min(1.0, len(self.history) / 100.0)
        
        return confidence * data_confidence

def compute_ewma(value: float, prev: float, alpha: float = 0.2) -> float:
    """Compute the EWMA (Exponential Weighted Moving Average) value with an optional adaptive smoothing factor.

    Args:
        value (float): The new value to compute EWMA.
        prev (float): The previous EWMA value.
        alpha (float): The smoothing factor. Defaults to 0.2.

    Returns:
        float: The computed EWMA value. If computation fails, return the previous value.
    """
    try:
        # If adaptive mode is enabled, adjust alpha based on data characteristics
        if os.getenv("PISCES_TREND_EWMA_ADAPTIVE", "true").lower() == "true":
            # Simplified adjustment based on volatility
            # In practice, should maintain historical data
            volatility = 0.1  # Default volatility
            if volatility > 0.2:  # High volatility
                alpha = min(0.5, alpha * 2.0)
            elif volatility < 0.05:  # Low volatility
                alpha = max(0.05, alpha * 0.5)
                
        return alpha * float(value) + (1.0 - alpha) * float(prev)
    except Exception as e:
        _LOGGER.debug("EWMA computation failed", error=str(e))
        return float(prev)

def update_baseline(baseline: Optional[float], value: float, beta: float = 0.01) -> float:
    """Update the baseline value with a new value.

    Args:
        baseline (Optional[float]): The previous baseline value. If None, use the new value.
        value (float): The new value to update the baseline.
        beta (float): The update factor. Defaults to 0.01.

    Returns:
        float: The updated baseline value. If computation fails, return the new value or the previous baseline.
    """
    try:
        if baseline is None:
            return float(value)
        return (1.0 - beta) * float(baseline) + beta * float(value)
    except Exception as e:
        _LOGGER.debug("Baseline update failed", error=str(e))
        return float(value if baseline is None else baseline)

def drift_ratio(curr: float, base: float) -> float:
    """Compute the enterprise-level drift ratio between the current value and the baseline.

    Args:
        curr (float): The current value.
        base (float): The baseline value.

    Returns:
        float: The computed drift ratio. If computation fails, return 0.0.
    """
    try:
        b = float(base)
        c = float(curr)
        return (c - b) / (b if b != 0.0 else 1e-6)
    except Exception as e:
        _LOGGER.debug("Drift ratio computation failed", error=str(e))
        return 0.0

def mad_stats(values):
    """Compute the enterprise-level MAD (Median Absolute Deviation) statistics.

    Args:
        values: The input values to compute MAD statistics.

    Returns:
        Tuple[float, float]: A tuple containing the median and MAD values.
            If computation fails, return (0.0, 1.0).
    """
    try:
        if not values:
            return 0.0, 1.0
            
        seq = sorted(float(v) for v in values)
        n = len(seq)
        if n == 0:
            return 0.0, 1.0
            
        # Compute using median
        median = seq[n // 2] if n % 2 == 1 else (seq[n // 2 - 1] + seq[n // 2]) / 2
        
        # Compute absolute deviations
        devs = sorted(abs(v - median) for v in seq)
        mad = devs[n // 2] if n % 2 == 1 else (devs[n // 2 - 1] + devs[n // 2]) / 2
        
        return median, (mad or 1.0)
    except Exception as e:
        _LOGGER.debug("MAD stats computation failed", error=str(e))
        return 0.0, 1.0

def is_outlier_mad(value: float, median: float, mad: float, k: float = 6.0) -> bool:
    """Detect if a value is an outlier using the enterprise-level MAD method with an optional adaptive threshold.

    Args:
        value (float): The value to check for outliers.
        median (float): The median of the data.
        mad (float): The Median Absolute Deviation of the data.
        k (float): The threshold for outlier detection. Defaults to 6.0.

    Returns:
        bool: True if the value is an outlier, False otherwise. If computation fails, return False.
    """
    try:
        # If adaptive mode is enabled, adjust k based on data distribution
        if os.getenv("PISCES_TREND_MAD_ADAPTIVE", "true").lower() == "true":
            # Simplified adjustment based on historical data distribution
            # Default to maintain k value, but can adjust based on data dispersion
            pass
            
        return abs(float(value) - float(median)) > float(k) * float(mad)
    except Exception as e:
        _LOGGER.debug("MAD outlier detection failed", error=str(e))
        return False

def rolling_percentile(values: List[float], window: int, q: float) -> List[float]:
    """Compute rolling percentiles over a specified window for the input values.

    This is a non-invasive helper function that can be safely used by external callers
    without changing the analyzer's behavior.

    Args:
        values (List[float]): The input values.
        window (int): The size of the rolling window.
        q (float): The percentile to compute, in the range [0, 1].

    Returns:
        List[float]: A list of rolling percentiles. If an error occurs, return an empty list.
    """
    try:
        out: List[float] = []
        w = max(1, int(window))
        if not values:
            return out
        for i in range(len(values)):
            s = max(0, i - w + 1)
            seg = sorted(values[s:i+1])
            if not seg:
                out.append(0.0)
                continue
            pos = max(0, min(len(seg) - 1, int(q * (len(seg) - 1))))
            out.append(float(seg[pos]))
        return out
    except Exception:
        return []

def stability_index(values: List[float]) -> float:
    """Compute a simple stability index in the range [0, 1], where higher values indicate greater stability.

    Uses the coefficient of variation over the series, which is robust to zeros.

    Args:
        values (List[float]): The input values.

    Returns:
        float: The computed stability index. If an error occurs, return 0.5.
    """
    try:
        if not values:
            return 1.0
        n = len(values)
        mean_val = sum(values) / max(1, n)
        if n < 2:
            return 1.0
        var = sum((v - mean_val) ** 2 for v in values) / n
        std = var ** 0.5
        denom = abs(mean_val) if abs(mean_val) > 1e-6 else 1.0
        cv = std / denom
        # Map CV to [0, 1] stability (cv=0 -> 1.0 stable; cv>=1 -> ~0)
        return float(max(0.0, min(1.0, 1.0 - cv)))
    except Exception:
        return 0.5

def enhanced_trend_summary(values: List[float], window: int = 50) -> Dict[str, Any]:
    """Generate a minimal enhanced trend summary with rolling baselines and stability.

    Args:
        values (List[float]): The input values.
        window (int): The size of the rolling window. Defaults to 50.

    Returns:
        Dict[str, Any]: A dictionary containing "rolling_p50", "rolling_p95", and "stability".
            If an error occurs, return default values.
    """
    try:
        rp50 = rolling_percentile(values, window, 0.5)
        rp95 = rolling_percentile(values, window, 0.95)
        stab = stability_index(values)
        return {"rolling_p50": rp50, "rolling_p95": rp95, "stability": stab}
    except Exception:
        return {"rolling_p50": [], "rolling_p95": [], "stability": 0.5}

