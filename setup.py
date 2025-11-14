from setuptools import setup, find_packages

setup(
    name="peft-lora-eval",
    version="0.1.0",
    packages=find_packages(),  # Auto-discovers src/ and subpackages
    install_requires=[],  # Dependencies handled via requirements-colab.txt
    author="Constantine Serkov",
    description="PEFT LoRA evaluation project",
    python_requires=">=3.10",
)