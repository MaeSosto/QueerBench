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
    response = response.replace(r" '", "")
    #response = lib.re.sub(r'.', '', response)
    response = response.lower()
    response = f"['{response}']" 
    return response

# The function `getListFromString` takes a string input, removes certain characters, splits the string
# by commas, and returns a list of the resulting elements.
def getListFromString(text):
    text = lib.re.sub(r"'", "", str(text))
    text = lib.re.sub(r'\]', '', text)
    text = lib.re.sub(r'\[', '', text)
    return list(map(str, text.split(",")))

def getCSVFile(folder, modelName, predictionsConsidered):
    files = []
    for f in lib.os.listdir(folder):
        pred = f.replace(f'{modelName}_', '').replace('.csv', '')
        try:
            if lib.re.match(modelName, f) and int(pred) >= predictionsConsidered:
                files.append(int(pred))
        except: 
            continue
    files.sort()
    try:
        return lib.pd.read_csv(f'{folder+modelName}_{files[0]}.csv')
    except Exception as X:
        print("EXC - There are no files related to the specified model [{modelName}] with at least {predictionsConsidered} words predicted")

