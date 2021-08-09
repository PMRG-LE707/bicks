import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bicks",
    version="0.1.0",
    author="cwxie",
    author_email="cw.xie@qq.com",
    description="find BICs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PMRG-LE707/bicks",
    project_urls={
        "Bug Tracker": "https://github.com/PMRG-LE707/bicks/issues",
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