from setuptools import find_packages, setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="corporate_emission_reports",
    version="0.0.3",
    packages=find_packages(),
    install_requires=requirements,
)

