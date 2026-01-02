from setuptools import setup, find_packages

setup(
    name='npp-accident-classifier',
    version='1.0.0',
    author='romanchaa997',
    description='LSTM multi-task classifier for NPP accident detection',
    url='https://github.com/romanchaa997/npp-accident-classifier',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'pydantic>=1.8.0',
        'fastapi>=0.68.0',
    ],
)
