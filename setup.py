from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyellipsoid',
    version='0.0.3',
    description='Ellipsoids drawing and analysis in 3D volumetric images.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/ashkarin/pyellipsoid',
    author='Andrei Shkarin',
    author_email='andrei.shkarin@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='ellipsoid drawing rotation analysis',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/ashkarin/pyellipsoid/issues',
        'Source': 'https://github.com/ashkarin/pyellipsoid'
    }
)
