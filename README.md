### A Statistical Model of Word Rank Evolution

**Quick Description:** This repository houses all of the Python code and data in order to replicate the figures and tables presented in the paper titled "A Statistical Model of Word Rank Evolution" by Alex John Quijano, Rick Dale, and Suzanne Sindi (2021).

**Getting Started and Dependencies**

```
git clone https://github.com/stressosaurus/a-statistical-model-of-word-rank-evolution.git
pip3 install --user -r requirements.txt
```

**Data Downloads**

* Please follow the instructions in the separate repository [raw-data-google-ngram](https://github.com/stressosaurus/raw-data-google-ngram) to download and pre-process the necessary Google Ngram Data.
* Make sure that all pre-processed data is in the directory `'/google-ngram/'+n+'gram-normalized/'+l+'/'` where `n=1` and `l` is a language code. For example `l="eng"`.

**Computations**

* Run the scripts below to post-process the google ngram data, perform WF inspired model simulations, and rank change computations.

```
bash google-1gram-data-computations-precompute.py
bash wright-fisher-simulations-precompute.py
bash rank-change-precompute.py
```

* *Step 2:* Run and follow the code in the Jupyter Notebook shown below.

```
jupyter notebook MAIN-1.ipynb
```