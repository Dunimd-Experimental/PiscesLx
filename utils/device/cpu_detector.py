#!/usr/bin/env python3

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

import psutil
import platform
import multiprocessing
from utils import PiscesLxCoreLog
from typing import Dict, Any, Optional

class PiscesLxCoreDeviceCpuDetector:
    """
    CPU detector with comprehensive hardware capabilities analysis.
    
    This class provides functionality to detect and analyze various CPU hardware 
    and performance metrics, and assess its capability for AI workloads.
    """
    
    def __init__(self) -> None:
        """
        Initialize the CPU detector with empty information containers and a logger.
        """
        self.cpu_info: Dict[str, Any] = {}  # Stores basic CPU information
        self.architecture_info: Dict[str, Any] = {}  # Stores CPU architecture details
        self.performance_metrics: Dict[str, Any] = {}  # Stores performance measurements
        self.logger = PiscesLxCoreLog()  # Logger instance for structured logging
        
    def detect(self) -> Dict[str, Any]:
        """
        Perform comprehensive CPU hardware and performance analysis.
        
        Executes multiple detection routines to collect CPU data including basic info,
        architecture details, performance metrics, etc., then evaluates the CPU's suitability
        for AI tasks.

        Returns:
            Dict[str, Any]: Dictionary with keys 'basic_info', 'architecture', 'performance',
                            and 'is_capable' containing respective collected data and evaluation.
        """
        self._detect_basic_info()
        self._detect_architecture_info()
        self._detect_performance_metrics()
        self._detect_vector_instructions()
        self._detect_memory_info()
        self._detect_thermal_info()
        
        return {
            'basic_info': self.cpu_info,
            'architecture': self.architecture_info,
            'performance': self.performance_metrics,
            'is_capable': self._assess_capability()
        }
    
    def _detect_basic_info(self) -> None:
        """
        Detect basic CPU information.
        
        Collects fundamental CPU characteristics like core counts, clock speeds, and platform identifiers.
        Falls back to default values if detection encounters errors.
        """
        try:
            logical_cores: Optional[int] = psutil.cpu_count(logical=True)
            physical_cores: Optional[int] = psutil.cpu_count(logical=False)
            
            self.cpu_info = {
                'physical_cores': physical_cores or logical_cores or 0,
                'logical_cores': logical_cores or 0,
                'threads_per_core': (logical_cores // physical_cores) if physical_cores and logical_cores else 1,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_freq': self._get_cpu_frequency(),
                'processor': platform.processor(),
                'machine': platform.machine(),
                'platform': platform.platform(),
                'system': platform.system(),
                'node': platform.node()
            }
            
        except Exception as e:
            self.logger.error("Failed to detect basic CPU info", error=str(e))
            self.cpu_info = {
                'physical_cores': multiprocessing.cpu_count(),
                'logical_cores': multiprocessing.cpu_count(),
                'threads_per_core': 1,
                'cpu_percent': 0,
                'cpu_freq': {'current': 0, 'min': 0, 'max': 0},
                'processor': platform.processor(),
                'machine': platform.machine(),
                'platform': platform.platform(),
                'system': platform.system(),
                'node': platform.node()
            }
    
    def _get_cpu_frequency(self) -> Dict[str, float]:
        """
        Get CPU frequency information.
        
        Attempts to fetch current, minimum, and maximum CPU frequencies.
        Provides zeroed defaults if fetching fails.

        Returns:
            Dict[str, float]: Contains 'current', 'min', and 'max' frequency values in MHz.
        """
        try:
            freq = psutil.cpu_freq()
            if freq:
                return {
                    'current': freq.current,
                    'min': freq.min or 0,
                    'max': freq.max or 0
                }
        except Exception as e:
            self.logger.debug(
                "get cpu_freq failed",
                event="CPU",
                message="get cpu_freq failed",
                error=str(e),
                error_class=type(e).__name__
            )
        return {'current': 0, 'min': 0, 'max': 0}
    
    def _detect_architecture_info(self) -> None:
        """
        Detect CPU architecture information.
        
        Uses the `cpuinfo` library to gather detailed architectural features.
        Reverts to platform-based information when the library is unavailable.
        """
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            
            self.architecture_info = {
                'vendor_id': cpu_info.get('vendor_id_raw', 'unknown'),
                'brand_raw': cpu_info.get('brand_raw', 'unknown'),
                'arch': cpu_info.get('arch_string_raw', 'unknown'),
                'bits': cpu_info.get('bits', 64),
                'family': cpu_info.get('family', 0),
                'model': cpu_info.get('model', 0),
                'stepping': cpu_info.get('stepping', 0),
                'flags': cpu_info.get('flags', []),
                'l1_data_cache_size': cpu_info.get('l1_data_cache_size', 0),
                'l1_instruction_cache_size': cpu_info.get('l1_instruction_cache_size', 0),
                'l2_cache_size': cpu_info.get('l2_cache_size', 0),
                'l3_cache_size': cpu_info.get('l3_cache_size', 0)
            }
            
        except ImportError:
            self.architecture_info = {
                'vendor_id': 'unknown',
                'brand_raw': platform.processor(),
                'arch': platform.machine(),
                'bits': 64 if platform.machine().endswith('64') else 32,
                'family': 0,
                'model': 0,
                'stepping': 0,
                'flags': [],
                'l1_data_cache_size': 0,
                'l1_instruction_cache_size': 0,
                'l2_cache_size': 0,
                'l3_cache_size': 0
            }
    
    def _detect_performance_metrics(self) -> None:
        """
        Detect CPU performance metrics.
        
        Measures computational throughput via synthetic benchmarks for both single-core
        and multi-core scenarios. Defaults are used on failure.
        """
        try:
            import time
            
            # Benchmark single-core performance
            start_time = time.time()
            _ = sum(i * i for i in range(1000000))  # Synthetic load
            single_core_time = time.time() - start_time
            
            # Benchmark multi-core performance
            start_time = time.time()
            try:
                with multiprocessing.Pool() as pool:
                    results = pool.map(
                        lambda x: sum(i * i for i in range(x, x + 100000)),
                        range(0, 1000000, 100000)
                    )
                    _ = sum(results)
            except Exception as e:
                self.logger.debug(
                    "multiprocessing pool unavailable, fallback to sequential",
                    event="CPU",
                    message="multiprocessing pool unavailable, fallback to sequential",
                    error=str(e),
                    error_class=type(e).__name__
                )
                _ = sum(i * i for i in range(1000000))  # Sequential fallback
            multi_core_time = time.time() - start_time
            
            self.performance_metrics = {
                'single_core_score': round(1.0 / single_core_time, 2),
                'multi_core_score': round(1.0 / multi_core_time, 2),
                'parallel_efficiency': round(multi_core_time / single_core_time, 2),
                'single_core_time': round(single_core_time, 4),
                'multi_core_time': round(multi_core_time, 4)
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to detect performance metrics",
                event="CPU",
                message="Failed to detect performance metrics",
                error=str(e)
            )
            self.performance_metrics = {
                'single_core_score': 0,
                'multi_core_score': 0,
                'parallel_efficiency': 1.0,
                'single_core_time': 0,
                'multi_core_time': 0
            }
    
    def _detect_vector_instructions(self) -> None:
        """
        Detect vector instruction support.
        
        Analyzes CPU feature flags to identify supported vector instruction sets and determines
        the highest SIMD level achievable.
        """
        flags = self.architecture_info.get('flags', [])
        
        vector_instructions = {
            'sse': any('sse' in flag.lower() for flag in flags),
            'sse2': 'sse2' in flags,
            'sse3': 'sse3' in flags,
            'sse4_1': 'sse4_1' in flags,
            'sse4_2': 'sse4_2' in flags,
            'avx': 'avx' in flags,
            'avx2': 'avx2' in flags,
            'avx512': any('avx512' in flag for flag in flags),
            'fma': 'fma' in flags,
            'neon': any('neon' in flag.lower() for flag in flags),
            'vmx': any('vmx' in flag.lower() for flag in flags)
        }
        
        self.architecture_info['vector_instructions'] = vector_instructions
        
        # Select highest SIMD level
        if vector_instructions['avx512']:
            simd_level = 'avx512'
        elif vector_instructions['avx2']:
            simd_level = 'avx2'
        elif vector_instructions['avx']:
            simd_level = 'avx'
        elif vector_instructions['sse4_2']:
            simd_level = 'sse4_2'
        else:
            simd_level = 'basic'
            
        self.architecture_info['simd_level'] = simd_level
    
    def _detect_memory_info(self) -> None:
        """
        Detect system memory information.
        
        Gathers virtual and swap memory statistics.
        Falls back to zeroed values on failure.
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            self.cpu_info['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percentage': memory.percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'swap_percentage': swap.percent
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to detect memory info",
                event="CPU",
                message="Failed to detect memory info",
                error=str(e)
            )
            self.cpu_info['memory'] = {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percentage': 0,
                'swap_total_gb': 0,
                'swap_used_gb': 0,
                'swap_percentage': 0
            }
    
    def _detect_thermal_info(self) -> Dict[str, Any]:
        """
        Detect thermal information.
        
        Attempts cross-platform retrieval of CPU temperatures and fan speeds.
        Returns an empty dict if no sensor data is accessible.

        Returns:
            Dict[str, Any]: Thermal sensor readings indexed by component name.
        """
        thermal_info: Dict[str, Any] = {}
        
        try:
            try:
                import psutil
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        thermal_info['cpu_temp'] = {}
                        for name, entries in temps.items():
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                thermal_info['cpu_temp'][name] = [
                                    {
                                        'label': entry.label,
                                        'current': entry.current,
                                        'high': entry.high,
                                        'critical': entry.critical
                                    }
                                    for entry in entries
                                ]
            except (ImportError, AttributeError, NotImplementedError, PermissionError) as e:
                self.logger.debug(
                    "thermal sensors unavailable",
                    event="CPU",
                    message="thermal sensors unavailable",
                    error=str(e),
                    error_class=type(e).__name__
                )
            
            try:
                import psutil
                if hasattr(psutil, 'sensors_fans'):
                    fans = psutil.sensors_fans()
                    if fans:
                        thermal_info['fans'] = {}
                        for name, entries in fans.items():
                            thermal_info['fans'][name] = [
                                {'label': entry.label, 'current': entry.current}
                                for entry in entries
                            ]
            except (ImportError, AttributeError, NotImplementedError, PermissionError) as e:
                self.logger.debug(
                    "fan sensors unavailable",
                    event="CPU",
                    message="fan sensors unavailable",
                    error=str(e),
                    error_class=type(e).__name__
                )
                            
        except Exception as e:
            self.logger.debug(
                "thermal info retrieval failed or unsupported",
                event="CPU",
                message="thermal info retrieval failed or unsupported",
                error=str(e),
                error_class=type(e).__name__
            )
        
        return thermal_info
    
    def _assess_capability(self) -> Dict[str, Any]:
        """
        Assess CPU capability for AI workloads.
        
        Computes a capability score based on physical cores, frequency, SIMD support,
        cache size, and available memory. Classifies the CPU into one of four capability levels.

        Returns:
            Dict[str, Any]: Evaluation result with 'level', 'score', 'recommendation',
                            and breakdown of contributing factors.
        """
        capability_score = 0.0
        
        # Core count contribution
        physical_cores = self.cpu_info.get('physical_cores', 0)
        if physical_cores >= 16:
            capability_score += 0.4
        elif physical_cores >= 8:
            capability_score += 0.3
        elif physical_cores >= 4:
            capability_score += 0.2
        else:
            capability_score += 0.1
        
        # Clock speed bonus
        freq = self.cpu_info.get('cpu_freq', {}).get('current', 0)
        if freq >= 3000:
            capability_score += 0.2
        elif freq >= 2000:
            capability_score += 0.1
        
        # SIMD instruction set advantage
        simd_level = self.architecture_info.get('simd_level', 'basic')
        if simd_level == 'avx512':
            capability_score += 0.3
        elif simd_level == 'avx2':
            capability_score += 0.2
        elif simd_level == 'avx':
            capability_score += 0.1
        
        # Cache capacity bonus
        l3_cache = self.architecture_info.get('l3_cache_size', 0)
        if l3_cache >= 32768:  # 32MB+
            capability_score += 0.1
        elif l3_cache >= 16384:  # 16MB+
            capability_score += 0.05
        
        # Memory bandwidth factor
        mem_gb = self.cpu_info.get('memory', {}).get('total_gb', 0)
        if mem_gb >= 32:  # 32GB+
            capability_score += 0.05
        elif mem_gb >= 16:  # 16GB+
            capability_score += 0.02
        
        # Capability classification
        if capability_score >= 0.8:
            capability_level = 'high'
            recommendation = 'Excellent for CPU inference and training'
        elif capability_score >= 0.6:
            capability_level = 'medium'
            recommendation = 'Good for CPU inference, limited training'
        elif capability_score >= 0.4:
            capability_level = 'low'
            recommendation = 'Basic CPU inference only'
        else:
            capability_level = 'minimal'
            recommendation = 'Not recommended for AI workloads'

        return {
            'level': capability_level,
            'score': round(min(capability_score, 1.0), 2),
            'recommendation': recommendation,
            'details': {
                'core_contribution': min(physical_cores / 32, 0.4),
                'frequency_contribution': 0.2 if freq >= 3000 else 0.1,
                'simd_contribution': 0.3 if simd_level == 'avx512' else 0.2 if simd_level == 'avx2' else 0.1,
                'cache_contribution': 0.1 if l3_cache >= 32768 else 0.05,
                'memory_contribution': 0.05 if mem_gb >= 32 else 0.02
            }
        }
    
    def get_recommended_strategy(self, model_size: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recommended strategy based on CPU capabilities.
        
        Constructs an execution plan optimized for the detected CPU profile and specified model size.

        Args:
            model_size (str, optional): Model scale indicator like "7B" or "70B". Defaults to None.

        Returns:
            Dict[str, Any]: Execution configuration including mode, thread allocation, reasoning,
                            precision settings, memory usage preferences, batch sizing, and optimizations.
        """
        capability = self._assess_capability()
        level = capability['level']
        
        # Evaluate how model size affects strategy
        model_impact = self._assess_model_impact(model_size)
        
        logical_cores = self.cpu_info.get('logical_cores', 4)
        
        if level == 'high':
            return {
                'mode': 'cpu_optimized',
                'threads': min(logical_cores, 8),  # Cap threads for optimal efficiency
                'reason': f'High-performance CPU: {capability["recommendation"]}',
                'mixed_precision': False,  # CPUs gain little from FP16
                'memory_efficient': model_impact['large_model'],
                'batch_size': 4 if not model_impact['large_model'] else 1,
                'optimization': 'avx512' if self.architecture_info.get('simd_level') == 'avx512' else 'avx2'
            }
        elif level == 'medium':
            return {
                'mode': 'cpu_standard',
                'threads': min(logical_cores, 4),
                'reason': f'Medium-performance CPU: {capability["recommendation"]}',
                'mixed_precision': False,
                'memory_efficient': True,
                'batch_size': 2 if not model_impact['large_model'] else 1,
                'optimization': self.architecture_info.get('simd_level', 'basic')
            }
        else:
            return {
                'mode': 'cpu_fallback',
                'threads': 2,  # Minimal thread allocation
                'reason': f'Limited CPU capability: {capability["recommendation"]}',
                'mixed_precision': False,
                'memory_efficient': True,
                'batch_size': 1,
                'optimization': 'basic'
            }
    
    def _assess_model_impact(self, model_size: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess model size impact on CPU strategy.
        
        Interprets the model size descriptor to determine whether it represents a large-scale model.

        Args:
            model_size (str, optional): Descriptor such as "7B" or "70B". Defaults to None.

        Returns:
            Dict[str, Any]: Indicates whether the model is large ('large_model') and its parameter count in billions ('model_params_b').
        """
        if not model_size:
            return {'large_model': False, 'model_params_b': 0}
            
        try:
            size_str = model_size.upper().replace('B', '')  # Normalize input
            params_b = float(size_str)
            
            return {
                'large_model': params_b >= 7,  # Models >=7B are large for CPU
                'model_params_b': params_b
            }
        except Exception as e:
            self.logger.debug(
                "parse model_size failed",
                event="CPU",
                message="parse model_size failed",
                error=str(e),
                error_class=type(e).__name__
            )
            return {'large_model': False, 'model_params_b': 0}
