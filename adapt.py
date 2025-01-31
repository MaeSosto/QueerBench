from LIB_constants import * 

def getCategory(templateFile, sentence):
    for index,row in templateFile.iterrows():
        if row.loc[TEMPLATE] == sentence:
            return row.loc[CATEGORY]
        
def adapt(model):
    predictionFile = pd.read_csv(OUTPUT_SENTENCES+model+".csv")
    templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH)
    dicSentences = {
        'index' : [],
        TEMPLATE: [],
        TYPE: [],
        CATEGORY: [],
        'subject': [],
        'prediction': [],
    }
    for index,row in tqdm(predictionFile.iterrows(), total=predictionFile.shape[0], desc=f'Evaluating {model}\'s answers', unit=' sentences'):
        dicSentences['index'].append(index)
        dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences[CATEGORY].append(getCategory(templateFile, row.loc[TEMPLATE]))
        dicSentences['subject'].append("")
        dicSentences['prediction'].append(f"['{re.sub(row.loc[TEMPLATE], '', row.loc[GENERATED].replace('.', ''))}']")
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs('adapted/', exist_ok=True)
        df.to_csv('adapted/'+model+".csv", sep=';', index=False)


models = [LLAMA3_70B, GPT4, GEMMA2_27B]
for m in models:
    adapt(m)
            