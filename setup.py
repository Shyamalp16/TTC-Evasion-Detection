"""
Setup script for SnitchSystem project.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="snitchsystem",
    version="1.0.0",
    author="SnitchSystem Team",
    author_email="team@snitchsystem.com",
    description="Privacy-focused fare evasion detection system for subway gates",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/snitchsystem",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Transportation",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies will be installed from requirements.txt files
    ],
    extras_require={
        "edge": [
            "opencv-python>=4.8.0",
            "ultralytics>=8.0.0",
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "loguru>=0.7.0",
            "pydantic>=2.0.0",
            "aiohttp>=3.8.0",
        ],
        "server": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "alembic>=1.10.0",
            "loguru>=0.7.0",
            "pydantic>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snitch-edge=edge.main:main",
            "snitch-server=server.run:main",
        ],
    },
)
