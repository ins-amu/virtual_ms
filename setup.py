# from sbi_nmms.__init__ import __version__ as v
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="vms",
    version="0.1",
    author="Abolfazl ziaeemehr",
    author_email="a.ziaeemehr@gmail.com",
    description="Virtual multiple sclerosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/Ziaeemehr/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: tested on Ubuntu 20.04",
    ],
    # python_requires='>=3.9',
    # packages=['vbi'],
    # package_dir={'vbi': 'vbi'},
    # package_data={'vbi': ['CPPModels/*.so']},
    # install_requires=requirements,
    # include_package_data=True,
)
