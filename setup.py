import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lensless_class",
    version="0.0.1",
    author="Eric Bezzam",
    author_email="ebezzam@gmail.com",
    description="Functions and scripts for lensless classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebezzam/LenslessClassification",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "click",
    ],
)
