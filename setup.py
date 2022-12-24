"""
Details about install of this package with setup.py:
* https://packaging.python.org/tutorials/packaging-projects/
"""

from setuptools import find_packages, setup

metadata = dict(
    name='minipeak',
    version='0.0.1',
    license='BSD-3',
    maintainer='jobesu14',
    maintainer_email='jobesu14@gmail.com',
    python_requires='==3.8.*',
    packages=find_packages(),
    # Use command below to see the output of find_packages()
    # python3 -c "from setuptools import setup, find_packages; print(find_packages())"
    platforms=['Linux', 'Windows'],
    install_requires=['pyabf',
                      'matplotlib>=3.2.1',
                      'mplcursors',
                      'openpyxl'
                      ],
    extras_require={
                   'dev': ['pep8-naming',
                           'flake8',
                           'mypy',
                           'pytest>=3.9'],
                   },
    entry_points={
        'console_scripts': [
            'plot_minis = minipeak.scripts.plot_minis:main'
        ]
    },
    zip_safe=False,
    include_package_data=True,
)

setup(**metadata)
