# FairnessNLP

## Usage

### SentDebias

We used the SentDebias code found on the [GitHub](https://github.com/pliang279/sent_debias) for [Liang et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.488.pdf). To use the Twitter dataset, we created a custom `TwitterProcessor` class for processing our data. Using `run_classifier.py`, we fine-tuned a biased and debiased version of BERT on the Twitter dataset according to the README on their repository.
