from lib.constants import *

def createTemplate():
    templateFile = pd.read_csv('dataset_source/template.csv')
    nounsFile = pd.read_csv(NOUNS_PATH)
    pronounsFile = pd.read_csv(PRONOUNS_PATH)
    dataList =[]
    print("๏ Generating template...")
    
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc='Creating template', unit=' sentences'):
        sentence = row.loc[TEMPLATE]
        dfSubjects = pd.concat([nounsFile, pronounsFile])
        
        #Creating sentences with nouns and pronouns
        for ind, r in dfSubjects.iterrows():
            init = True if sentence.split(TARGET_)[0] == '' else False
            the = ('The ' if init else 'the ') if r.loc[THE] == 'y' else ('My ' if init else 'my ')
            _subject =  the + f"{r.loc[SUBJECT]}" + f'{" person" if r.loc[PERSON] == "y" else ""}'
            _sentence = sentence.replace(TARGET_, _subject)
            
            dataList.append([
                _sentence, #new template
                r.loc[TYPE], #type
                r.loc[CATEGORY], #category
                r.loc[SUBJECT] #subject
            ]) 
    data_df = pd.DataFrame(dataList, columns=["template", TYPE, CATEGORY, SUBJECT])
    data_df.to_csv(TEMPLATE_PATH_COMPLETE, index_label = 'index')
    print("๏ File template generated!")

createTemplate()
