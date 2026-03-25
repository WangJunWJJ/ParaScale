from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="parascale",
    version="0.2.0",
    description=(
        "A PyTorch-based deep learning framework supporting "
        "multiple parallelism strategies"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ParaScale Team",
    author_email="parascale@example.com",
    url="https://github.com/parascale/parascale",
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*"]
    ),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "torchvision>=0.10.0",
            "pytest>=6.0.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.900",
        ],
        "examples": [
            "torchvision>=0.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords=(
        "deep learning, parallel training, pytorch, "
        "distributed computing, tensor parallel, "
        "pipeline parallel, data parallel"
    ),
    license="MIT License",
    project_urls={
        "Bug Tracker": "https://github.com/parascale/parascale/issues",
        "Documentation": "https://parascale.readthedocs.io",
        "Source Code": "https://github.com/parascale/parascale",
    },
)
