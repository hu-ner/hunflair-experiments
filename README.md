# HunFlair experiments
This Repository contains the source code to reproduce the experiments conducted in [Weber et al. 
"HunFlair: An Easy-to-Use Tool for State-of-the-Art Biomedical Named Entity Recognition"]().

## Usage
To reproduce the experiments the following steps have to be performed:
* Setup environment and install necessary packages / models:
~~~
pip install -r requirements.txt

chmod u+x install_scispacy_models.sh
./install_scispacy_models.sh
~~~
* Download and prepare the evaluation corpora
~~~
python download_and_prepare_corpora.py 
~~~
* Run experiments
~~~
chmod u+x run_experiments.sh
./run_experiments.sh
~~~

<sub>Note for the competitors HUNER and MISC we provide the annotations in `annotations`.</sub>

## Citing HunFlair
Please cite the following paper when using HunFlair:
~~~
@article{weber2020hunflair,
  author    = {Weber, Leon and S{\"a}nger, Mario and M{\"u}nchmeyer, Jannes and Habibi, Maryam and Leser, Ulf and Akbik, Alan},
  title     = {HunFlair: An Easy-to-Use Tool for State-of-the-ArtBiomedical Named Entity Recognition},
  journal   = {},
  volume    = {},
  year      = {2020},
  ee        = {http://arxiv.org/abs/XXX}
}
~~~






