import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dilated-attention",
    version="0.1",
    description="",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Boris Albar',
    author_email = 'b.albar@catie.fr',
    url="https://github.com/catie-aq/dilated-attention/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
    ]
)
