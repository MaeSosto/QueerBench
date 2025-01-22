#Imports
import API
import torch
from transformers import AutoModel, BertTokenizer, BertForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
import logging
import pandas as pd
from tqdm import tqdm
from afinn import Afinn
import unidecode
import time
import os
import re
import seaborn as sb 
from time import sleep
from googleapiclient import discovery
import numpy as np
np.random.seed(42)
import statistics as st
import math
import ast
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.pyplot import savefig
from matplotlib.colors import LinearSegmentedColormap


logging.basicConfig(level=logging.INFO)# OPTIONAL
print(f"PyTorch version: {torch.__version__}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# # Global Variables
#Data Source
DATA_SOURCE = 'dataset_source/'
OUTPUT_TEMPLATE = 'output_template/'
OUTPUT_EVALUATIONS = 'output_evaluations/'
OUTPUT_QUEERBENCH = 'output_queerbench/'
OUTPUT_GRAPHS = 'output_graphs/'
OUTPUT_PREDICTION = 'output_prediction/'
TEMPLATE_PATH = DATA_SOURCE + 'template.csv'
NOUNS_PATH = DATA_SOURCE + 'nouns.csv'
PRONOUNS_PATH = DATA_SOURCE + 'pronouns.csv'
TEMPLATES_COMPLETE_PATH = OUTPUT_TEMPLATE + 'template_complete.csv'

# TEMPLATE MAP
BE_ = '<be>'
TARGET_ = '<target>'
HAVE_ = '<have>'
WERE_ = '<were>'
QUEERNESS = 'queerness'
TYPE = 'type'
CATEGORY= 'category'
SUBJECT = 'subject'
MASKBERT_ = '\[MASK\]'
MASKBERT = '[MASK]'
MASKROBERT = '<mask>'
THE = 'the'

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
MODELS = {
    'BERT_base': 'bert-base-uncased',
    'BERT_large': 'bert-large-uncased',
    'RoBERTa_base': 'roberta-base',
    'RoBERTa_large': 'roberta-large',
    'AlBERT_base': 'albert-base-v2',
    'AlBERT_large': 'albert-large-v2',
    'BERTweet_base': 'vinai/bertweet-base',
    'BERTweet_large': 'vinai/bertweet-large',
    # LLAMA3 : 'llama3',
    # LLAMA3_70B : 'llama3:70b',
    # GEMMA2 : 'gemma2',
    # GEMMA2_27B : 'gemma2:27b',
    # GPT4 : 'gpt-4o'
}

BERT_BASE = 'BERT_base'
BERT_LARGE = 'BERT_large'
ROBERTA_BASE = 'RoBERTa_base'
ROBERTA_LARGE = 'RoBERTa_large'
ALBERT_BASE = 'AlBERT_base'
ALBERT_LARGE = 'AlBERT_large'
BERTTWEET_BASE = 'BERTweet_base'
BERTTWEET_LARGE = 'BERTweet_large'

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
PRONOUN_CATEGORIES = [BINARY, NEO, NEUTRAL]
NOUN_TYPES = [QUEER, NONQUEER]
NOUN_CATEGORIES = [ORIENTATION, IDENTITY, OTHER]

#Evaluation Tools
AFINN = 'AFINN'
HURTLEX = 'HurtLex'
PERSPECTIVE = 'Perspective'

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


# ## Utils
def strip_accent(accented_string):
    return unidecode.unidecode(str(accented_string))

def hurtLexSetup():
    language = 'en'
    hurtlex = pd.read_csv(f"https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/{language.upper()}/1.2/hurtlex_{language.upper()}.tsv", sep="\t")
    hurtlex = hurtlex[hurtlex["level"] == "conservative"]
    hurtlex["lemma"] = hurtlex["lemma"].apply(strip_accent)
    return hurtlex

def perspectiveSetup():
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API.API_KEY_PERSPECTIVE,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    return client

def truncate(float_number, decimal_places = 2):
    multiplier = 10 ** decimal_places
    try:
        return int(float_number * multiplier) / multiplier
    except:
        return 0
    
def shiftAfinnScore(num):
    # Calculates the distance from 0 and map it in a range between [0, 100]
    return abs(num) * 20