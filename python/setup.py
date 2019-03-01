from __future__ import print_function
from setuptools import setup, find_packages
import sys
import io

setup(
    name="pyspark-iforest",
    version="2.4.0",
    author="Titicaca",
    author_email="lake_titicaca@outlook.com",
    description="PySpark Wrapper for Spark-IForest",
    long_description=io.open("README.md", encoding="UTF-8").read(),
    license="MIT",
    url="https://github.com/titicaca/spark-iforest/python",
    packages=find_packages(),
    entry_points={
    },
    data_files=[#('data', []),
                ('doc', ['README.md']),
               ],
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: NLP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
            'pyspark==2.4.0'
        ],
    zip_safe=True,
)
