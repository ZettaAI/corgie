import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corgie",
    version="0.0.9",
    author="Sergiy Popovych",
    author_email="sergiy.popovich@gmail.com",
    description="Connectomics Registration General Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supersergiy/corgie",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
      'sphinx-click',
      'torchfields',
      'torch',
      'artificery',
      'gevent',
      'torchvision',
      'numpy',
      'six',
      'pyyaml',
      'mazepa',
      'click-option-group',
      'click>=7,<8',
      'procspec',
      'modelhouse',
      'idna>=2.5,<3',
      'google-auth<2.0dev,>=1.11.0',
      'cloud-volume<3',
      'scikit-image',
      'h5py',
      'kimimaro',
      'docutils==0.15.0'
    ],
    entry_points={
        "console_scripts": [
            "corgie = corgie.main:cli",
            "corgie-worker = corgie.worker:worker",
        ],
    },
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
