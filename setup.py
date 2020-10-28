import setuptools


description = open("README.md").read()
version = open("VERSION").read()
requirements = open("requirements.txt").read()

setuptools.setup(
    name="argmining",
    version=version,
    author="Dimov Ilya",
    description="Argumentation mining thesis.",
    long_description=description,
    long_description_conttype="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    zip_safe=False
)
