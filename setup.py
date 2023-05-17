from setuptools import setup, find_packages

setup(
  name = 'pifold-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'Unofficial implementation of PiFold in PyTorch.',
  author = 'Dohoon Lee',
  author_email = 'dohlee.bioinfo@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/dohlee/pifold-pytorch',
  keywords = [
    'artificial intelligence',
    'bioinformatics',
    'computational biology',
  ],
  install_requires=[
    'einops>=0.3',
    'numpy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)