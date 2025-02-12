from lib.constants import *
import os 
import re
import pandas as pd


def getCSVFile(folder, modelName, predictionsConsidered):
    files = []
    for f in os.listdir(folder):
        pred = f.replace(f'{modelName}_', '').replace('.csv', '')
        try:
            if re.match(modelName, f) and int(pred) >= predictionsConsidered:
                files.append(int(pred))
        except: 
            continue
    files.sort()
    try:
        return pd.read_csv(f'{folder+modelName}_{files[0]}.csv')
    except Exception as X:
        print("EXC - There are no files related to the specified model [{modelName}] with at least {predictionsConsidered} words predicted")

def getNounsCat(template, type):
    nounsFile = pd.read_csv(NOUNS_PATH)
    for _, row in nounsFile.iterrows():
        rowType = row.loc[TYPE]
        rowSubj =  row.loc[SUBJECT]
        if rowType == type and rowSubj in template:
            return row.loc[CATEGORY]
    return ""
    
folder = OUTPUT_EVALUATION
predictionsConsidered = 1
model = BERT_BASE
MODEL_LIST = [BERT_BASE, BERT_LARGE, ROBERTA_BASE, ROBERTA_LARGE, ALBERT_BASE, ALBERT_LARGE, BERTTWEET_BASE, BERTTWEET_LARGE, LLAMA3, LLAMA3_70B, GEMMA2]

for model in MODEL_LIST:
    categories = []
    types = []
    templateFile = getCSVFile(folder, model, predictionsConsidered)
    for _,row in tqdm(templateFile.iterrows(), total = templateFile.shape[0]):
        types.append(row.loc[TYPE])
        if row.loc[TYPE] == PRONOUN:
            if 'they' in row.loc[TEMPLATE]:
                categories.append(NEUTRAL)
            elif 'he' in row.loc[TEMPLATE] or 'she' in row.loc[TEMPLATE]:
                categories.append(BINARY)
            else:
                categories.append(NEO)
        else:
            categories.append(getNounsCat(row.loc[TEMPLATE], row.loc[TYPE]))
    templateFile[CATEGORY] = categories
    templateFile.to_csv(f"{OUTPUT_EVALUATION+ model}_1.csv")