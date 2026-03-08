from setuptools import setup, find_packages

setup(
    name="enron-memory-graph",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'google-generativeai>=0.3.0',
        'flask>=2.0.0',
        'tqdm>=4.62.0',
        'numpy>=1.21.0',
        'python-dateutil>=2.8.2'
    ],
    author="Your Name",
    description="Memory graph system for Enron emails",
    python_requires=">=3.8",
)
