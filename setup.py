from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='MultiArmedBandits',
    description='Assignment 1 of Reinforcement Learning Practical at University of Groningen',
    install_requires=requirements,
)
