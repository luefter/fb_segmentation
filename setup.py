import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fbg",
    version="0.0.1",
    description="A package for computing foreground-background segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    author="Caspar & Sinans",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Prof. Barbara Hammer",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10.4",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="segmentation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=["pathlib"],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8", "pre-commit"],
        "test": ["pytest"],
    },
    package_data={
        "sample": ["package_data.dat"],
    },
    data_files=[("my_data", ["data/data_file"])],
    entry_points={
        "console_scripts": [
            "sample=sample:main",
        ],
    },
    project_urls={
        "Source": "https://github.com/luefter/fb_segmentation",
    },
)
