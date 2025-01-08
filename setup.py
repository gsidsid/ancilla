from setuptools import setup, find_packages

setup(
    name="ancilla",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "polygon-api-client>=1.12.0",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "pytz",
        "scipy>=1.9.0",
        "fredapi>=0.5.2",
        "plotly>=5.24.1"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ]
    },
)
