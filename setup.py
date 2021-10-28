from setuptools import setup, find_packages

setup(
    version='2.2.3',
    name='calib3d',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/ispgroupucl/calib3d",
    licence="LGPL",
    python_requires='>=3.6',
    description="Python 3D calibration and homogenous coordinates computation library",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python"
    ],
)
