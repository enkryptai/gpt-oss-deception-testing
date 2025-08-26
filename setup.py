#!/usr/bin/env python3
"""
Setup script for GPT-OSS Deception Testing Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpt-oss-deception-testing",
    version="1.0.0",
    author="Enkrypt AI",
    author_email="nitin@enkryptai.com",
    description="Framework for testing OpenAI GPT-OSS-20B deception capabilities across high-stakes scenarios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/enkryptai/gpt-oss-deception-testing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "local": [
            # vLLM requires special installation - see README
            # "vllm==0.10.1+gptoss",  # Manual installation required
        ],
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "all": [
            # Includes development dependencies
            "pytest>=6.0", 
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "deception-test=comprehensive_deception_test:main",
        ],
    },
    zip_safe=False,
)
