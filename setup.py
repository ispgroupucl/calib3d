from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    version="2.8.0",
    name='calib3d',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/ispgroupucl/calib3d",
    license="LGPL",
    python_requires='>=3.6',
    description="Python 3D calibration and homogenous coordinates computation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python"
    ],
    extras_requires={
        "tensorflow": ["tensorflow>=2.4"],
        "pycuda": ["pycuda"],
        "matplotlib": ["matplotlib"],
    }
)
