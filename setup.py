import os

from setuptools import setup

BASE_VERSION = '1.7.0'  # update regardless whether you update keras or pytorch or both.
FRAMEWORK = os.getenv('FRAMEWORK', 'keras')  # keras, pytorch.

# common packages.
INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'matplotlib'
]

if FRAMEWORK == 'keras':
    LIB_PACKAGE = ['nbeats_keras']
    INSTALL_REQUIRES.extend([
        'keras',
        'tensorflow'
    ])

elif FRAMEWORK == 'pytorch':
    LIB_PACKAGE = ['nbeats_pytorch']
    INSTALL_REQUIRES.extend([
        'torch',
    ])
else:
    raise ValueError('Unknown framework.')

setup(
    name=f'nbeats-{FRAMEWORK}',
    version=BASE_VERSION,
    description='N-Beats',
    author='Philippe Remy (Pytorch), Jean Sebastien Dhr (Keras)',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=LIB_PACKAGE,
    install_requires=INSTALL_REQUIRES
)
