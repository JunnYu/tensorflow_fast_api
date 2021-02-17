from setuptools import setup, find_packages

setup(
    name='tf_fast_api',
    packages=find_packages(),
    version='0.0.1',
    license='MIT',
    description='tf fast api',
    author='Jun Yu',
    author_email='573009727@qq.com',
    url='https://github.com/JunnYu/tensorflow_fast_api',
    keywords=['tf'],
    install_requires=['tensorflow>=2.3.0', 'fastcore'],
)