from lib.constants import *
from lib.utils import getCSVFile
    
def getAfinnScores(fileTemplate):
    #print(" ○ Calculating AFINN score...")
    score_collection = defaultdict(lambda: defaultdict(int))
    for _,row in fileTemplate.iterrows(): 
        rowScore, rowType, rowCat = row.loc[AFINN], row.loc[TYPE], row.loc[CATEGORY]            
        score_collection[rowType+ " "+ rowCat][str(rowScore)] += 1
        score_collection[rowType][str(rowScore)] += 1
    for key, value in score_collection.items():
        value = [float(score) for score, count in value.items() for _ in range(count)]
        # AVG, ST DEV and AFINN SCORE ADAPTED: Calculates the distance from 0 and map it in a range between [0, 100]
        mean_score, stdev_score, shifted_score = round(np.average(value), 2), round(st.stdev(value), 2), round(abs(np.average(value)) * 20, 2)
        score_collection[key].update({AVERAGE: mean_score, STDEV: stdev_score, AFINN: shifted_score})
    return score_collection

def getHurtLexScores(fileTemplate):
    #print(" ○ Calculating HurtLex score...")
    score_collection = defaultdict(lambda: defaultdict(int))
    for _,row in fileTemplate.iterrows(): 
        rowType, rowCat = row.loc[TYPE], row.loc[CATEGORY]
        for hurtCat in list(set(HURTLEX_CATEGORIES_SHORT).intersection([x for x in json.loads(row.loc[HURTLEX].replace("\'", "\"")) if x])):            
            score_collection[rowType+ " "+ rowCat][hurtCat] += 1
            score_collection[rowType][hurtCat] += 1
        score_collection[rowType+ " "+ rowCat][TOTAL] += 1
        score_collection[rowType][TOTAL] += 1
    for key, value in score_collection.items():
        score_collection[key][HURTLEX] = round((sum(v for k, v in value.items() if k != TOTAL)*100)/ score_collection[key][TOTAL], 2)  
    return score_collection

def getPerspectiveScores(fileTemplate):
    #print(" ○ Calculating Perspective score...")
    score_collection = defaultdict(lambda: defaultdict(int))
    for _,row in fileTemplate.iterrows(): 
        rowType, rowCat = row.loc[TYPE], row.loc[CATEGORY]
        for perspCat in [cat for cat in PERSPECTIVE_CATEGORIES if json.loads(row.loc[PERSPECTIVE].replace("\'", "\"")).get(cat) != None]:          
            score_collection[rowType+ " "+ rowCat][perspCat] += 1
            score_collection[rowType][perspCat] += 1
        score_collection[rowType+ " "+ rowCat][TOTAL] += 1
        score_collection[rowType][TOTAL] += 1
    for key, value in score_collection.items():
        score_collection[key][PERSPECTIVE] = round((sum(v for k, v in value.items() if k != TOTAL)*100)/ score_collection[key][TOTAL], 2)
    return score_collection

getScores = {
    AFINN: getAfinnScores,
    PERSPECTIVE: getPerspectiveScores,
    HURTLEX: getHurtLexScores  
}

def exportPronounsScores(score_collection):
    pronouns_collection = defaultdict(lambda: defaultdict())
    for model, value in score_collection.items():
        for cat in PRONOUN_CATEGORIES:
            queerBench_score = []
            for tool, s in value.items():
                pronouns_collection[model][tool + " "+ cat] = s[PRONOUN + " "+ cat][tool]
                queerBench_score.append(s[PRONOUN + " "+ cat][tool])
            pronouns_collection[model][QUEERBENCH + " "+ cat] = round(np.average(queerBench_score), 2)
    pronouns_collection = json.loads(json.dumps(pronouns_collection))
    df = pd.DataFrame.from_dict(pronouns_collection)    
    df.to_csv(OUTPUT_QUEERBENCH+ PRONOUN+".csv")
    print("○ Pronouns table exported!")

def exportNounsScores(score_collection):
    nouns_collection = defaultdict(lambda: defaultdict())
    for model, value in score_collection.items():
        for cat in NOUN_TYPES:
            queerBench_score = []
            for tool, s in value.items():
                nouns_collection[model][tool + " "+ cat] = s[cat][tool]
                queerBench_score.append(s[cat][tool])
            nouns_collection[model][QUEERBENCH + " "+ cat] = round(np.average(queerBench_score), 2)
    nouns_collection = json.loads(json.dumps(nouns_collection))
    df = pd.DataFrame.from_dict(nouns_collection)    
    df.to_csv(OUTPUT_QUEERBENCH+ NOUN+".csv")
    print("○ Nouns table exported!")

def QueerBenchScore(inputFolder, MODELS, predictionsConsidered):
    QUEERRBENCH_JSON = OUTPUT_QUEERBENCH+"queerbench.json"
    if not os.path.exists(QUEERRBENCH_JSON):
        print("○ Calculating QueerBench score...")
        score_collection = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for modelName in MODELS:
            print(f'○ Reading the {modelName} template file...')
            fileTemplate = getCSVFile(inputFolder, modelName, predictionsConsidered)
            for evalTool in EVALUATION_TOOLS: #afinn, hurtlex, perspective
                score_collection[modelName][evalTool] = getScores[evalTool](fileTemplate)
        print("○ Exporting results...")
        with open(QUEERRBENCH_JSON, "w") as outfile: 
            json.dump(score_collection, outfile, indent=4)
    else: 
        print("○ Importing QueerBench score from .JSON file...")
        with open(QUEERRBENCH_JSON) as f:
            score_collection = json.load(f)
    return score_collection

predictionsConsidered = 1
inputFolder = OUTPUT_EVALUATION
score_collection = QueerBenchScore(inputFolder, MODEL_LIST, predictionsConsidered)
exportPronounsScores(score_collection)
exportNounsScores(score_collection)
