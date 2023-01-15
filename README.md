conseillify-research
==============================

### Recommendation engine for spotify million playlist challange

#### Downloading data requires
- Valid aicrowd account
- Kaggle.json file in ~./kaggle
- at least 30GB of free memory

#### To be able to train the models it is necessary to use a machine that has at least 8 GB of RAM and 16 GB of free memory.


Project Organization
------------

    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make data` or `make train`
    ├── README.md           <- The top-level README for developers using this project.
    ├── artifacts           <- To be able to build a container, copy into this directory all of the training outputs.
    ├── data
    │   ├── dataset         <- Lightfm dataset object artifacts.
    │   ├── external        <- Data from third party sources.
    │   ├── predictions     <- The resulted recommendations after each model component.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    │   ├── candidate_gen   <- Collaborative candidate generation model.
    │   ├── coldstart       <- HDBSCAN based coldstart model.
    │   ├── coldstart_birch <- BIRCH based coldstart model.
    │   ├── diversification <- Hybrid model based on Graph Convolutional Network.
    │   └── learn_to_rank   <- Ranking model based on hybrid matrix factorization with lightfm.
    │   └── reranking       <- Reranking model based on python reranking library.
    ├── notebooks          <- Jupyter notebooks, that had been executed on kaggle, used for 
    |                            exploratory data analysis and pulling track features
    │
    ├── reports            <- Generated analysis files
    │   └── eval_coldstart <- Results of testing coldstart model in collaborative scenarios
    │   └── eval_gen_cand  <- Evaluations of purely collaborative model         
    │   └── eval_r_a_d     <- Evaluations of the full architecture
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── serving            <- This directory contains the scripts necessary to build a docker container for serving the model
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    |   │   ├── candidate_gen   <- Collaborative candidate generation model.
    |   │   ├── coldstart       <- HDBSCAN based coldstart model.
    |   │   ├── coldstart_birch <- BIRCH based coldstart model.
    |   │   ├── diversification <- Hybrid model based on Graph Convolutional Network.
    |   │   └── learn_to_rank   <- Ranking model based on hybrid matrix factorization with lightfm.
    |   │   └── reranking       <- Reranking model based on python reranking library.

#### Most important make instructions:
- make requirements - installs libraries
- make data - download all the datasets from aicrowd and kaggle
- make features - processes the data fr training
- make generate_candidates - trains a collaborative model
- make rank - trains a hybrid model
- make diversify - trains a GCN 
- make coldstart_birch - trains a coldstart model based on BIRCH algorithm
