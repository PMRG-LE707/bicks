import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicky",
    version="0.0.2",
    author="cwxie",
    author_email="541361425@qq.com",
    description="find BICs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PMRG-LE707/findbic",
    project_urls={
        "Bug Tracker": "https://github.com/PMRG-LE707/findbic/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./"),
    python_requires=">=3.6",
)