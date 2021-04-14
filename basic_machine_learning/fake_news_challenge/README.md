# Modifications to the baseline FNC implementation

Information about the fake news challenge can be found on [FakeChallenge.org](http://fakenewschallenge.org).

This repository contains the baseline FNC implementation by:
* Byron Galbraith (Github: @bgalbraith, Slack: @byron)
* Humza Iqbal (GitHub: @humzaiqbal, Slack: @humza)
* HJ van Veen (GitHub/Slack: @mlwave)
* Delip Rao (GitHub: @delip, Slack: @dr)
* James Thorne (GitHub/Slack: @j6mes)
* Yuxi Pan (GitHub: @yuxip, Slack: @yuxipan)

with code that reads the dataset, extracts some simple features, trains and cross-validated model and performs an evaluation on a hold-out set of data.

It also is being worked on currently in two areas:

1. Adding new features

2. Adding new classification methods

## Questions / Issues
For questions about the Fake News Challenge or the original repo, go to the slack group [fakenewschallenge.slack.com](https://fakenewschallenge.slack.com)

For questions about our repo, send an email to Elliot Greenlee at egreenle@vols.utk.edu or Patricia Eckhart at pdraney@vols.utk.edu.

## Getting Started
Currently we are using a MacBook Pro (Retina, 15-inch, Mid 2015) running macOS Sierra Version 10.12.3. Programming is done using the terminal application and PyCharm 2016.2.3. This isn't a distribution so we won't troubleshoot your issues, but we know how annoying setup can be so hopefully this will help.

* Clone the repository

    ``git clone the_repo_name``

* Open the directory as a project using PyCharm
* Set up a python 3.6 virtual environment using PyCharm, ideally in a directory adjacent to your repo clone, or create your own virtual environment
* Activate the virtual environment in the terminal

    ``source your_virtual_environment_name/bin/activate``

* Update pip using 

    ``sudo pip install --upgrade pip``

* Navigate to the repo directory and install the requirements using 

    ``sudo pip install -r requirements.txt``

* Download the nltk database by running nltk_download.py. A window will open, and you should choose the directory where the download should occur, and then click download. One of the files, panlex_lite, is large and may have errors, but once it gets to that point you should be ok to cancel. This script came from stack overflow, and no guarantees about it are made with regards to safety (I ran it on my computer though)

    ``python nltk_download.py``

* Run fnc_kfold.py. It took about 25 minutes to run and should look like 

    ``python fnc_kfold.py``

    ```
    Reading dataset
    Total stances: 49972
    Total bodies: 1683
    9622it [00:40, 238.66it/s]
    ...
    Iter  Train Loss  Remaining Time 
    1  35396.3298  39.86s
    ...
    Score: 3538.0 out of 4448.5	(79.53242666067214%)
    ```

## Useful functions
### dataset class
The dataset class reads the FNC-1 dataset and loads the stances and article bodies into two separate containers.

    dataset = DataSet()

You can access these through the ``.stances`` and ``.articles`` variables

    print("Total stances: " + str(len(dataset.stances)))
    print("Total article bodies: " + str(len(dataset.articles)))

* ``.articles`` is a dictionary of articles, indexed by the body id. For example, the text from the 144th article can be printed with the following command:
   ``print(dataset.articles[144])``

### Hold-out set split
Data is split using the ``generate_hold_out_split()`` function. This function ensures that the article bodies between the training set are not present in the hold-out set. This accepts the following arguments. The body IDs are written to disk.

* ``dataset`` - a dataset class that contains the articles and bodies
* ``training=0.8`` - the percentage of data used for the training set (``1-training`` is used for the hold-out set)
* ``base_dir="splits/"``- the directory in which the ids are to be written to disk


### k-fold split
The training set is split into ``k`` folds using the ``kfold_split`` function. This reads the holdout/training split from the disk and generates it if the split is not present.

* ``dataset`` - dataset reader
* ``training = 0.8`` - passed to the hold-out split generation function
* ``n_folds = 10`` - number of folds
* ``base_dir="splits"`` - directory to read dataset splits from or write to

This returns 2 items: a array of arrays that contain the ids for stances for each fold, an array that contains the holdout stance IDs.

### Getting headline/stance from IDs
The ``get_stances_for_folds`` function returns the stances from the original dataset. See ``fnc_kfold.py`` for example usage.



## Scoring Your Classifier

The ``report_score`` function in ``utils/score.py`` is based off the original scorer provided in the FNC-1 dataset repository written by @bgalbraith.

``report_score`` expects 2 parameters. A list of actual stances (i.e. from the dev dataset), and a list of predicted stances (i.e. what you classifier predicts on the dev dataset). In addition to computing the score, it will also print the score as a percentage of the max score given any set of gold-standard data (such as from a  fold or from the hold-out set).

    predicted = ['unrelated','discuss',...]
    actual = [stance['Stance'] for stance in holdout_stances]

    report_score(actual, predicted)

This will print a confusion matrix and a final score your classifier. We provide the scores for a classifier with a simple set of features which you should be able to match and eventually beat!

|               | agree         | disagree      | discuss       | unrelated     |
|-----------    |-------        |----------     |---------      |-----------    |
|   agree       |    118        |     3         |    556        |    85         |
| disagree      |    14         |     3         |    130        |    15         |
|  discuss      |    58         |     5         |   1527        |    210        |
| unrelated     |     5         |     1         |    98         |   6794        |
Score: 3538.0 out of 4448.5	(79.53%)
