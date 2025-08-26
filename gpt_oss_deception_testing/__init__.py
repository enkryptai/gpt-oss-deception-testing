"""
GPT-OSS Deception Testing Framework

A comprehensive framework for testing OpenAI GPT-OSS-20B's deception capabilities 
across high-stakes scenarios with statistical rigor.
"""

__version__ = "1.0.0"
__author__ = "Enkrypt AI"
__email__ = "nitin@enkryptai.com"

from .comprehensive_deception_test import (
    DeceptionTester,
    TestConfiguration, 
    TestResult,
    CellSummary,
    run_comprehensive_evaluation,
    load_test_config,
    generate_test_configurations
)

__all__ = [
    "DeceptionTester",
    "TestConfiguration",
    "TestResult", 
    "CellSummary",
    "run_comprehensive_evaluation",
    "load_test_config",
    "generate_test_configurations"
]
