from setuptools import setup, find_packages

setup(
    name="flash-kmeans",
    version="0.1.0",
    description="Fast batched K-Means clustering with Triton GPU kernels",
    author="Your Name",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "triton>=2.0",  # NVIDIA Triton language package
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
