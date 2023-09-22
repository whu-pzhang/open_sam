import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(name='open_sam',
          version='v0.1',
          description='Open SAM',
          long_description='',
          long_description_content_type='text/markdown',
          author='Zhang Pan',
          author_email='',
          keywords='computer vision, segmentation anything',
          url='',
          packages=find_packages(exclude=('configs', 'tools', 'demo')),
          include_package_data=True,
          classifiers=[
              'Development Status :: 4 - Beta',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ],
          license='Apache License 2.0',
          install_requires='',
          extras_require={},
          ext_modules=[],
          zip_safe=False)
