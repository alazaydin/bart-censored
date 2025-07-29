# The Censored Model for BART

This notebook implements the model for Coon and Lee (2022) with `pymc`.

The authors share data and code on [OSF](https://osf.io/3n98r/).

Suggested use:
1. download the OSF data and put in a folder "../coon_osf_files"
2. install `conda`
3. create environment
    - install [`pymc`](https://www.pymc.io/projects/docs/en/stable/installation.html)
    - or with conda use `environment.yml` (macOS): `conda env create -f environment.yml`
4. activate environment and install libraries: run `pip install -e .` after changing directory to project folder

Coon, J., & Lee, M. D. (2022). A Bayesian method for measuring risk propensity in the Balloon Analogue Risk Task. Behavior Research Methods, 54(2), 1010-1026.