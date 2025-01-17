from constants import *

def plural_form(be, sentence):
    if be == 'are':
        word = sentence.split(" ")[1]
        if word[-1] == 's':
            sentence = sentence.replace(word, word[:-1])
    return sentence

# # Build Complete Template
def templateBuilder():
    template = pd.read_csv(TEMPLATE_PATH, sep=";", dtype={'template': 'category', 'category': 'category'})
    nouns = pd.read_csv(NOUNS_PATH, sep=';')
    pronouns = pd.read_csv(PRONOUNS_PATH, sep=';')
    dataList =[]
    for index,row in tqdm(template.iterrows(), total=template.shape[0], desc='Creating template', unit=' sentences'):
        sentence = row.loc['template']
        
        #Creating sentences with nouns
        for ind, r in nouns.iterrows():
            _sentence = sentence.replace(TARGET_, f"The {r.loc[SUBJECT]} person") if r.loc[THE] == 'y' else sentence.replace(TARGET_, f"The {r.loc[SUBJECT]}")
            _sentence = _sentence.replace(BE_, 'is').replace(WERE_, 'was').replace(HAVE_, 'has')

            data=[
                _sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY], #category
                r.loc[SUBJECT] #subject
            ]
            dataList.append(data) 

        #Creating sentences with pronouns
        for ind, r in pronouns.iterrows():
            _sentence= plural_form(r.loc[BE_], sentence.replace(TARGET_, r.loc[SUBJECT]))
            _sentence = _sentence.replace(BE_, r.loc[BE_]).replace(WERE_, r.loc[WERE_]).replace(HAVE_, r.loc[HAVE_])

            data=[
                _sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY], #category
                r.loc[SUBJECT] #subject
            ]
            dataList.append(data) 
    data_df = pd.DataFrame(dataList, columns=["template", TYPE, CATEGORY, SUBJECT])
    print(data_df)
    os.makedirs(OUTPUT_TEMPLATE, exist_ok=True)
    data_df.to_csv(TEMPLATES_COMPLETE_PATH, sep=';', index=False)
   

# # Main
#Create the complete template
templateBuilder()
