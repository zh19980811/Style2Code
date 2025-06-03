from setuptools import setup, find_packages

setup(
    name='code-style-evaluator',
    version='0.1',
    packages=find_packages(),
    install_requires=['matplotlib'],
    author='Your Name',
    description='A tool to evaluate Python code naming style and function quality.',
    url='https://github.com/your-username/code-style-evaluator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
