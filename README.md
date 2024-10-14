# A Bragg Pulse Simulator for He3-He4



## Dev env setup
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

By default, I want to create the environment in the project directory `./` so it doesn't clutter my home directory. But if you want to store packages for this environment in the conda default directory, use `--name py311_he34sim` instead of `--prefix`.  

`macos-armos` uses the Apple Accelerate framework, `intel` uses MKL for hardware accelerated linear algebra computations.

Don't forget to activate the newly created environment.

VSCode's jupyter page should auto detect the new environment. 


```bash
conda activate ./envs/py311_he34sim 
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








Random command logs needed to get shit working 
```bash
which ffmpeg
# and then update os.environ["IMAGEIO_FFMPEG_EXE"]

conda install matplotlib --update-deps --force-reinstall

conda install opencv --update-deps --force-reinstall

conda install numpy --update-deps --force-reinstall

conda install 'numpy>=2.0' --update-deps --force-reinstall

```