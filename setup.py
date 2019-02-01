from setuptools import setup, find_packages

setup(
    name='oceansar',
    version='19.01.01',
    # packages=['oceansar'],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    # package_dir={'': 'oceansar'},
    url='https://github.com/pakodekker/OCEANSAR',
    license='GPL-3.0',
    author='Paco Lopez Dekker',
    author_email='F.LopezDekker@tudelft.nl',
    description='',
    install_requires=['numpy', 'scipy', 'numexpr', 'numba', 'mpi4py', 'matplotlib', 'NetCDF4']
)
