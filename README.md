<!-- # A Bragg Pulse Simulator for He3-He4 -->
<!-- # Split-Step-Operator Method Schrödinger Equation Propagator  -->
<!-- Split-Time Evolution Propagator for the Schrödinger Equation -->
<!-- Schrödinger Time Evolution Propagator using Split-step operator method -->
# $\mathbb{STEPS}$
<!-- # $\textbf{S}\text{chr\"odiner}$ $\textbf{T}\text{ime}$ $\textbf{E}\text{volution}$ $\textbf{P}\text{ropagator}$ $\text{using}$ $\textbf{S}\text{plit-step}$ $\text{operator}$ $\text{method}$  -->
<!-- Split-sTep opErator method for Propagating Schrödinger equation -->
$\textbf{S}\text{plit-s}\textbf{T}\text{ep}$ 
$\text{op}\textbf{E}\text{rator}$ 
$\text{method}$ 
$\text{for}$
$\textbf{P}\text{ropagating}$
$\textbf{S}\text{chrödinger}$
$\textbf{E}\text{quation}$

[![python](https://img.shields.io/badge/python-3.12-gray.svg?style=flat&logo=python&logoColor=white&labelColor=black)](https://docs.python.org/3/whatsnew/3.12.html)
![numpy](https://img.shields.io/badge/numpy-black.svg?logo=numpy&logoColor=white)
![numba](https://img.shields.io/badge/numba-black.svg?logo=numba&logoColor=white)
![scipy](https://img.shields.io/badge/scipy-black.svg?logo=scipy&logoColor=white)
![conda](https://img.shields.io/badge/conda-black.svg?logo=anaconda&logoColor=white)


## Introduction
todo


## Dev env setup (Apr 2025)
As of April 2025, I've updated to use python 3.12 on macOS 15.4 arm64, and `miniconda` to manage the python packages for this project. Lower python versions should still work.  
```bash
conda env create --name steps python=3.12
conda activate steps
```
Note the environment name `steps` can be changed to whatever you like.

Then use `conda install package_name` to manually install the list of packages in `requirements.txt` or `env-hist-noarch.yml` file. 
Note some packages may not be available with `conda`, in that case, you can use `pip install package_name` to install them (some may not be available with `pip` either, in that case, you'll need `brew install package_name` or `apt-get install package_name`).


## Dev env setup (OLD Pre-2024)
I use [`miniconda`](https://docs.anaconda.com/miniconda/) to manage the python packages for this project, please have it installed in order to follow this setup guide.
I am using python 3.11 (as of Oct 2024) since this is the only version that somewhat satisfies all dependencies. At some point, I would like to move to python 3.12 to utilise the new typing features.
I'm also using VSCode for multi view, extensions, copilot, etc. For an identical dev experience, consider using the VSCode profile `py311_h34sim.code-profile`, and workspace `py311_h34sim.code-workspace`

**Create the conda environment**

Please run one of the following command.


```bash
conda env create --file env-hist-macos-arm64.yml --prefix ./envs/py311_he34sim
conda env create --file env-hist-intel.yml --prefix ./envs/py311_he34sim
conda env create --file env-hist-noarch.yml --prefix ./envs/py311_he34sim
```

```bash
conda env create --name py311_steps --file env-hist-macos-arm64.yml
conda env create --name py311_steps --file env-hist-intel.yml
conda env create --name py311_steps --file env-hist-noarch.yml
```

By default, I want to create the environment in the project directory `./` so it doesn't clutter my home directory. But if you want to store packages for this environment in the conda default directory, use `--name py311_he34sim` instead of `--prefix`.  

`macos-armos` uses the Apple Accelerate framework, `intel` uses MKL for hardware accelerated linear algebra computations.

Don't forget to activate the newly created environment.

VSCode's jupyter page should auto detect the new environment. 


```bash
conda activate ./envs/py311_he34sim 
```

```bash
conda activate py311_he34sim 
```


and deactivate once done


```bash
conda deactivate 
```

To see all your current `conda` environments: 

```bash 
conda env list 
```

you should see `*` with path to this repo

**Update the conda environment configuration file** 

```bash
conda env export --from-history > env-hist.yml
```

and manually remove the `name` and `prefix` fields (? maybe it's fine without removing them? since I explicitly set the environment's name already), then I like to manually update the `macos-arm64` and `intel` configurations.

To export a list of packages actually installed, run: 


```bash
conda env export > env.yml
```

**Remove/Delete/Uninstall this conda environment**

```bash
conda env remove --prefix ./envs/py311_he34sim
```

## Git and jupyter

(Why do I have  `*.ipynb` in `.gitignore`.?)

Because jupyter's file is rather large with too much not-git-useful data, I am using [`jupytext`](https://jupytext.readthedocs.io/en/latest/using-cli.html) to convert `.ipynb` files to `.py` for better git readability. 

```bash
jupytext --to py twoParticle_v.a.6.ipynb
```

```bash
jupytext --to ipynb twoParticle_v.a.6.py
```

Because I'm using VSCode so I have to do these manual conversions, if you use jupyter notebook or lab, you can setup a pairing between `.ipynb` and `.py` files, and then can convert automatically, see doc [here](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html). 


Generate an `html` archives of the jupyter notebook 
```bash
jupyter nbconvert --to html twoParticle_v.a.6.ipynb
```




Some random command logs needed to get shit working 
```bash
which ffmpeg
# and then update os.environ["IMAGEIO_FFMPEG_EXE"]

conda install matplotlib --update-deps --force-reinstall

conda install opencv --update-deps --force-reinstall

conda install numpy --update-deps --force-reinstall

conda install 'numpy>=2.0' --update-deps --force-reinstall

```



### Using venv? *BAD IDEA DON'T DO THIS*
```bash
python -m venv venv
# For UNIX
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

```bash
sudo apt-get update
sudo apt-get install ffmpeg libblas-dev liblapack-dev libfftw3-dev libopencv-dev

brew install ffmpeg openblas fftw opencv
```



## Papers produced using this codebase 
[![arXiv](https://img.shields.io/badge/arXiv-2411.08356-dd3333.svg?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2411.08356)
```
@misc{yan2025proposalbelltestentangled,
      title={Proposal for a Bell Test with Entangled Atoms of Different Mass}, 
      author={X. T. Yan and S. Kannan and Y. S. Athreya and A. G. Truscott and S. S. Hodgman},
      year={2025},
      eprint={2411.08356},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2411.08356}, 
}
```



## Running a simulation sequence He3-He4

First initialise all functions and variable definitions with (🔄)

Then run 

1. Initialise initial state (🈶) section and check the plots look okay 
2. One test run  (🈶) , this is to get numba caching working, and get a runtime estimate 
    - tip: use benchmark tool to find 


## Running Bragg pulse simulations

TODO












