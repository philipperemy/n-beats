import os
import platform

from setuptools import setup
from pathlib import Path

BASE_VERSION = '1.8.0'  # update regardless whether you update keras or pytorch or both.
FRAMEWORK = os.getenv('FRAMEWORK', 'keras')  # keras, pytorch.
if Path('.torch').exists():
    FRAMEWORK = 'pytorch'

# common packages.
INSTALL_REQUIRES = [
    'numpy',
    'keract',  # for the intermediate outputs
    'pandas',
    'matplotlib',
    'protobuf<=3.20.2'
]

M1_MAC = platform.system() == 'Darwin' and platform.processor() == 'arm'
if M1_MAC:
    tensorflow = 'tensorflow-macos'
    # https://github.com/grpc/grpc/issues/25082
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'
    os.environ['GRPC_PYTHON_BUILD_SYSTEM_ZLIB'] = '1'
else:
    tensorflow = 'tensorflow'

if FRAMEWORK == 'keras':
    LIB_PACKAGE = ['nbeats_keras']
    INSTALL_REQUIRES.extend(['keras', tensorflow])

elif FRAMEWORK == 'pytorch':
    LIB_PACKAGE = ['nbeats_pytorch']
    INSTALL_REQUIRES.extend(['torch'])
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
