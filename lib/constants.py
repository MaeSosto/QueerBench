#Imports
import torch
import logging
import pandas as pd
from tqdm import tqdm
from googleapiclient import discovery
import unidecode
import os
import requests
import time
import re
from collections import defaultdict
import json
import numpy as np
import statistics as st
# import API
# import time
# import seaborn as sb 
# from time import sleep
# np.random.seed(42)
# import math
# import ast
# import matplotlib.pyplot as plt
# from matplotlib.transforms import Affine2D
# from matplotlib.pyplot import savefig
# from matplotlib.colors import LinearSegmentedColormap

logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)# OPTIONAL
print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# # Global Variables
#Data Source
DATA_SOURCE = 'dataset_source/'
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'
PRONOUNS_PATH = DATA_SOURCE + 'pronouns.csv'
TEMPLATE_PATH_COMPLETE = DATA_SOURCE + 'template_complete.csv'
OUTPUT_TEMPLATE = 'output_template/'
OUTPUT_QUEERBENCH = 'output_queerbench/'
OUTPUT_GRAPHS = 'output_graphs/'
OUTPUT_PREDICTION = 'output_prediction/'
OUTPUT_SENTENCES = 'output_sentences/'
OUTPUT_EVALUATION = 'output_evaluation/'

# TEMPLATE MAP
BE_ = '<be>'
TARGET_ = '<target>'
HAVE_ = '<have>'
WERE_ = '<were>'
QUEERNESS = 'queerness'
TYPE = 'type'
CATEGORY= 'category'
SUBJECT = 'subject'
MASKBERT_ = r'\[MASK\]'
MASKBERT = '[MASK]'
MASKROBERT = '<mask>'
THE = 'the'
PERSON = 'person'
TEMPLATE = 'template'
GENERATED = 'generated'

LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GPT4 = 'gpt-4o'

LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GPT4 = 'gpt-4o'

# MODELS
BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
ALBERT_BASE = 'AlBERT_base'
ALBERT_LARGE = 'AlBERT_large'
BERTTWEET_BASE = 'BERTweet_base'
BERTTWEET_LARGE = 'BERTweet_large'
LLAMA3 = 'llama3'
LLAMA3_70B = 'llama3:70b'
LLAMA3_3 = 'llama3.3'
GEMMA2 = 'gemma2'
GEMMA2_27B = 'gemma2:27b'
GEMINI_FLASH = "gemini-1.5-flash"
GPT4_MINI = 'gpt-4o-mini'
GPT4 = 'gpt-4o'
MODEL_LIST = [BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, ALBERT_BASE, ALBERT_LARGE, BERTTWEET_BASE, BERTTWEET_LARGE, LLAMA3, LLAMA3_70B, GEMMA2, GEMMA2_27B, GPT4_MINI]

MODEL_NAME = {
    BERT_BASE: 'bert-base-uncased',
    BERT_LARGE: 'bert-large-uncased',
    ROBERTA_BASE: 'roberta-base',
    ROBERTA_LARGE: 'roberta-large',
    ALBERT_BASE: 'albert-base-v2',
    ALBERT_LARGE: 'albert-large-v2',
    BERTTWEET_BASE: 'vinai/bertweet-base',
    BERTTWEET_LARGE: 'vinai/bertweet-large',
    # LLAMA3 : 'llama3',
    # LLAMA3_70B : 'llama3:70b',
    # GEMMA2 : 'gemma2',
    # GEMMA2_27B : 'gemma2:27b',
    # GPT4 : 'gpt-4o'
}

# TEMPLATE MAP
NOUN = 'noun'
PRONOUN = 'pronoun'
CATEGORY = 'category'
NAME = 'name'
TYPE = 'type'
BINARY = 'binary'
NEO = 'neo'
NEUTRAL = 'neutral'
QUEER = 'queer'
NONQUEER = 'non-queer'
ORIENTATION = 'orientation'
IDENTITY = 'identity'
OTHER = 'other'
SUBJECT_TYPE = [NOUN, PRONOUN]
PRONOUN_CATEGORIES = [BINARY, NEO, NEUTRAL]
NOUN_TYPES = [QUEER, NONQUEER]
NOUN_CATEGORIES = [ORIENTATION, IDENTITY, OTHER]

#Evaluation Tools
AFINN = 'AFINN'
HURTLEX = 'HurtLex'
PERSPECTIVE = 'Perspective'
EVALUATION_TOOLS = [AFINN, HURTLEX, PERSPECTIVE]

# HurtLex
HURTLEX_CATEGORIES = ['Animals', 'Crime and immoral behavior', 'Derogatory words', 'Female genitalia', 'Male genitalia', 'Homosexuality', 'Potential negative connotations', 'Professions and occupations', 'Prostitution', 'Social and economic disadvantage']
HURTLEX_CATEGORIES_SHORT = ['an', 're', 'cds', 'asf', 'asm', 'om', 'qas', 'pa', 'pr', 'is']

#Perspective
TOXICITY = 'TOXICITY'
INSULT = 'INSULT'
IDENTITY_ATTACK = 'IDENTITY_ATTACK'
PROFANITY = 'PROFANITY'
THREAT = 'THREAT'
PERSPECTIVE_CATEGORIES =[TOXICITY, INSULT, IDENTITY_ATTACK, PROFANITY, THREAT]

#Table utils
Y_AXE = ['Binary','Neutral', 'Neo', 'Queer Identity', 'Queer Orientation', 'Queer Other', 'Non-queer Identity', 'Non-queer Orientation', 'Non-queer Other', 'Queer', 'Non-queer']
