from setuptools import setup, find_packages

setup(
    name="flux",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "einops",
        "fire",
        "invisible-watermark",
        "safetensors"
    ]
) 