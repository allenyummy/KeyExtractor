from setuptools import setup

setup(
    name="keyextractor",
    packages=["keyextractor"],  # Chose the same as "name"
    version="0.1.0",
    license="MIT",
    description="KeyExtractor performs keyword extraction for chinese documents with state-of-the-art transformer models.",
    author="Yu-Lun Chiang",
    author_email="chiangyulun0914@gmail.com",
    url="https://github.com/allenyummy/KeyExtractor.git",
    download_url="https://github.com/allenyummy/KeyExtractor/archive/v_01.tar.gz",
    keywords=[
        "Natural Language Processing",
        "Keyword Extraction",
    ],
    install_requires=[
        "python==3.8",
        "torch>=1.7.1",
        "flair>=0.8.0post1",
        "transformers>=4.5.0",
        "ckip-transformers>=0.2.3",
        "pytest>=6.2.3",
        "pytest-mock>=3.5.1",
        "pytest-cov>=2.11.1",
        "black>=20.8b1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)