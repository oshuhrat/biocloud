from setuptools import setup

setup(
    name="biocloud",
    version="1.0.0",
    description="Bistable lattice model of bioelectric "
                "cancer cloud normalization with "
                "instructor cells",
    author="Ulugbek Khamidov",
    author_email="u.khamidov@alet.uz",
    url="https://github.com/YOUR_USERNAME/biocloud",
    py_modules=["biocloud"],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
)
