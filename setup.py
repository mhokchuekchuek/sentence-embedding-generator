from setuptools import find_packages, setup


def read_requirements(path: str) -> list:
    with open(path, "r") as fh:
        return fh.read().splitlines()


base_requirements = read_requirements("requirements.txt")

with open("readme.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="sentence-embedding-generator",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["embedding_generator.*"]),
    python_requires=">=3.9",
    install_requires=base_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Sentence Embedding Generator",
    ],
)
