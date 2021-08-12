# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

render_reqs = ['Pillow>=7.0.0']


setuptools.setup(
    name='selfcontact',
    version='0.2.0',
    packages=['selfcontact','selfcontact.utils', 'selfcontact.losses', 'selfcontact.fitting'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='PyTorch module to detect self-contact and self-interpenetration.',
    long_description=long_description,
    python_requires='>=3.6.0',
    author='Lea Mueller',
    author_email='lea.mueller@tuebingen.mpg.de',
    install_requires=[
        'torch>=1.0.1',
        'numpy>=1.18.1',
        'trimesh>=3.5.16'
    ],
    extra_require={
        'render': render_reqs
    }
)
