import lib.constants as lib

# This Python function allows the user to choose a model from a list of options provided by the
# `lib.MODEL_LIST` and returns the index of the chosen model.
# :return: The function `chooseModel()` returns the index of the chosen model from the
# `lib.MODEL_LIST`.
def chooseModel():
    chosenModel = -1
    while chosenModel < 0 or chosenModel > len(lib.MODEL_LIST)-1:
        print('๏ Select a model: ')
        for idx, x in enumerate(lib.MODEL_LIST):
            print(f"[{idx}] -  {x}")
        chosenModel = int(input())
    return chosenModel

# The `clean_response` function removes newline characters, double quotes, and backticks from a given
# response string.
def clean_response(response):
    response = lib.re.sub(r'\n', '', response)
    response = lib.re.sub(r'\"', '', response)
    response = lib.re.sub(r'`', '', response)
    response = response.replace('.', '')
    #response = lib.re.sub(r'.', '', response)
    response = response.lower()
    response = f"['{response}']" 
    return response

def strip_accent(accented_string):
    return lib.unidecode.unidecode(str(accented_string))

def hurtLexSetup():
    language = 'en'
    hurtlex = lib.pd.read_csv(f"https://raw.githubusercontent.com/MilaNLProc/hurtlex/master/lexica/{language.upper()}/1.2/hurtlex_{language.upper()}.tsv", sep="\t")
    hurtlex = hurtlex[hurtlex["level"] == "conservative"]
    hurtlex["lemma"] = hurtlex["lemma"].apply(strip_accent)
    return hurtlex

def perspectiveSetup():
    client = lib.discovery.build(
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