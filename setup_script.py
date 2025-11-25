# gd_cdss/setup.py

"""
Setup script for GD-CDSS package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gd-cdss",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gene-Drug Clinical Decision Support System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gd-cdss",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gd-cdss=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gd_cdss": ["artifacts/*"],
    },
)
