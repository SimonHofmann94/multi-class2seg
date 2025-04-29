from setuptools import setup, find_packages
setup(
    name="crackseg",
    version="0.0.0",
    package_dir={"": "class2seg"},   # <<< hier wichtig
    packages=find_packages("class2seg"),
)
