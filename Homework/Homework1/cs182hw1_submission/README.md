In this assignment you will practice writing backpropagation code and training
Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**

## Setup
Make sure your machine is set up with the assignment dependencies.

**[Option 1] Use Google Colab (Recommended):**
The preferred approach to do this assignment is to use [Google Colab](https://colab.research.google.com/).


**[Option 2] Use a local Conda environment:**
Another approach for installing all the assignment dependencies is to use
[Anaconda](https://www.continuum.io/downloads), which is a Python distribution
that includes many of the most popular Python packages for science, math,
engineering and data analysis. Once you install it you can skip all mentions of
requirements and you are ready to go directly to working on the assignment.

```bash
conda create -n cs182hw1 python=3.8 jupyter
conda activate cs182hw1
python -m pip install numpy==1.21.6 imageio==2.9.0 matplotlib==3.2.2
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.

```bash
cd deeplearning/datasets
./get_datasets.sh
cd ../..
```

If you are on Mac, this script may not work if you do not have the wget command
installed, but you can use curl instead with the alternative script.
```bash
cd deeplearning/datasets
./get_datasets_curl.sh
cd ../..
```

**Start Jupyter:**
After you have the CIFAR-10 data, you should start the IPython notebook server
from this directory.
```bash
jupyter notebook
```

### Fully-connected Neural Network
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

If you use Colab for this notebook, make sure to manually download the completed
notebook and place it in the assignment directory before submitting. Also remember
to download required output file and place it into submission_logs/ directory.
