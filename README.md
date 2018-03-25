# Toxic Comment Classification Challenge

The code for Kaggle competition https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

I won a silver medal with rank 106/4551 (top 3%)


# About the code
The code is implemented in Keras, with GPU tensorflow backend. Necessary packages must be installed 

Modeling data are from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

I used four pre-trained word embeddings. 

fastText crawl-300d-2M.vec. https://github.com/facebookresearch/fastText/blob/master/docs/english-vectors.md


glove crawl 300d, glove twitter 300d, glove wiki 300d, https://nlp.stanford.edu/projects/glove/

All the input files and model parameters are controlled by config.json. Before running the code, the config.json must be modified appropriately.

# Data cleaning
The major effort was spent on data cleaning! Since this is an sentiment classification task. Those words with strong emotion are important indicators. Unfortunately misspelling are everywhere, therefore, I put a lot of effort in spelling check, especially for those malicious words, and it worked! My ROCAUC scores were boosted with cleaned data. 

The code is in Data_cleaning.py. Cleaning the data before running are strongly recommended.

# Run the code
Again, the config.json must be properly modified to run the code. The file is quite elf explanatory

python Fit_predict.py

