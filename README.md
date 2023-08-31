# A RST Discourse Parser using neural network

We build up a RST discourse parser described in [Neural-based RST Parsing And Analysis In Persuasive Discourse]. 

Due to the licence of RST data corpus, we can't include the data in our project folder. To reproduce the result in the paper, you need to download it from the LDC. Put the TRAINING and TEST folders under data directory

### Usage:
1. Prepare Virtual Environment

    >conda create -n rst_env python=3.6
                                   
    >conda activate rst_env
                                   
    >pip install requirements.txt
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
2. Generate .merge file:

    please check [https://github.com/JinfenLi/NLP_discourse_ML](https://github.com/JinfenLi/NLP_discourse_ML) for preprocessing
    
3. run NN/main.py to train and predict

### Requirements:

All the codes are tested under Python 3.6. And see requirements.txt for python library dependency and you can install them with pip.


