from setuptools import setup, find_packages

setup(
    name='calib3d',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/ispgroupucl/calib3d",
    licence="LGPL",
    python_requires='>=3.6',
    description="Python 3D calibration and homogenous coordinates computation library",
    version='2.2.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python"
    ],
)
