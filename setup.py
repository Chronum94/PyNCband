from setuptools import setup

setup(
    name="pyncband",
    version="0.10.4",
    packages=["pyncband"],
    url="",
    license="MIT",
    author="Anubhab Haldar",
    author_email="",
    description="A 1st-order quantum dot bandgap calculator",
    install_requires=[
    "matplotlib>=2.0.0",
    "numpy>=1.14.5",
    "numba>=0.39.0",
    "scipy>=1.1.0",
    "typing>=3.6.4"
    ],
)
