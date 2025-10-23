#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'maxinforl_jax @ git+https://github.com/sukhijab/maxinforl_jax.git',
    'pandas',
    'jaxtyping',
    'numpy==1.26.4',
    'gymnasium==0.29.1',
    'distrax',
    'metaworld @git+https://github.com/sukhijab/Metaworld.git',
    'humanoid-bench @ git+https://github.com/sukhijab/humanoid-bench.git'
]

extras = {}
setup(
    name='combrl',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    # python_requires='< 3.10',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)

# pip install jaxlib==0.3.10 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# pip install jax==0.3.10 -f https://storage.googleapis.com/jax-releases/jax_releases.html
