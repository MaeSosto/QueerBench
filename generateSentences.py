from lib.constants import *
from lib.utils import *
import lib.API as API
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoModel, BertTokenizer, BertForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, AlbertTokenizer, AlbertForMaskedLM
URL_OLLAMA_LOCAL = "http://localhost:11434/api/generate"

def preExistingFile(modelName, numPrediction):
    filePath = f'{OUTPUT_SENTENCES+modelName}_{numPrediction}.csv'
    startingFrom, dicSentences = 0, {
        TYPE: [],
        TEMPLATE: [],
        GENERATED: [],
        CATEGORY: []
    }
    
    #If the file exists already in the output folder then take that one   
    if os.path.exists(filePath):
        df = pd.read_csv(filePath)
        startingFrom = df.shape[0]
        print(f"๏ Importing sentences from a pre-existing file [{startingFrom} sentences imported]")
        for idx, row in df.iterrows():
            dicSentences[TYPE].append(row.loc[TYPE])
            dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
            dicSentences[GENERATED].append(row.loc[GENERATED])
            dicSentences[CATEGORY].append(row.loc[CATEGORY])
        print("๏ Sentences imported correctly!")
    else:
        print("๏ Starting from the the source files")  
    return startingFrom, dicSentences

def initializeGemini(modelName = None):
    genai.configure(api_key=API.GENAI_API_KEY)
    return genai.GenerativeModel(GEMINI_FLASH), None

def initializeGPT(modelName = None):
    return OpenAI(api_key=API.OPENAI_API_KEY), None

def initializeBERT(modelName):
    val = MODEL_NAME[modelName]
    return BertForMaskedLM.from_pretrained(val), BertTokenizer.from_pretrained(val)

def initializeRoBERTa(modelName):
    return RobertaForMaskedLM.from_pretrained(MODEL_NAME[modelName]), RobertaTokenizer.from_pretrained(MODEL_NAME[modelName])

def initializeAlBERT(modelName):
    return AlbertForMaskedLM.from_pretrained(MODEL_NAME[modelName]), AlbertTokenizer.from_pretrained(MODEL_NAME[modelName])

def initializeBERTweet(modelName):
    return AutoModel.from_pretrained(MODEL_NAME[modelName]), AutoTokenizer.from_pretrained(MODEL_NAME[modelName], use_fast=False)

initialize_models = {
    BERT_BASE: initializeBERT,
    BERT_LARGE: initializeBERT,
    ALBERT_BASE: initializeAlBERT,
    ALBERT_LARGE: initializeAlBERT,
    ROBERTA_BASE: initializeRoBERTa,
    ROBERTA_LARGE: initializeRoBERTa,
    BERTTWEET_BASE: initializeBERTweet,
    BERTTWEET_LARGE: initializeBERTweet,
    GEMINI_FLASH: initializeGemini,
    GPT4: initializeGPT,
    GPT4_MINI: initializeGPT
}

def ollamaRequest (prompt, modelName, model = None, tokenizer = None, sentence = None, numPrediction = 1):
    response = requests.post(URL_OLLAMA_LOCAL, headers={
        "Content-Type": 'application/json'
    }, json={
        "model": modelName,
        "prompt": prompt,
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ],
        "options":{
            "temperature":0
        },
        "stream": False
    })
    return clean_response(response.json()['response'])

def geminiRequest(prompt, modelName, model, tokenizer = None, sentence = None, numPrediction = 1):
    resp =  clean_response(model.generate_content(prompt).text)
    time.sleep(2.5)
    return list(resp.lower())

def GPTRequest(prompt, modelName, model, tokenizer = None, sentence = None, numPrediction = 1):
    completion = model.chat.completions.create(
        model= modelName,
        store=True,
        messages=[{
            "role": "user", 
            "content": prompt
        }],
        temperature = 0   
        )
    response = completion.choices[0].message.content
    return clean_response(response)

def BERTRequest(prompt, modelName, model, tokenizer, sentence, numPrediction):
    sentence = "[CLS] %s [SEP]"%sentence
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        output = model(tokens_tensor)
        predictions = output[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, numPrediction, sorted=True)

    predictionList = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if ((modelName == ALBERT_BASE) or (modelName == ALBERT_LARGE)):
            predicted_token = predicted_token.replace(r'▁', '')
        predictionList.append(predicted_token)
    return predictionList  

def RoBERTaRequest(prompt, modelName, model, tokenizer, sentence, numPrediction):
    sentence = sentence.replace(MASKBERT, MASKROBERT)
    sentence = "<s> %s </s>"%sentence
    tokenized_text = tokenizer.tokenize(sentence)
    masked_index = tokenized_text.index(MASKROBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        output = model(tokens_tensor)
        predictions = output[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, numPrediction, sorted=True)

    predictionList = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        predictionList.append(predicted_token.replace('Ġ', ''))
    return predictionList  

request_models = {
    GEMINI_FLASH: geminiRequest,
    GPT4: GPTRequest,
    GPT4_MINI: GPTRequest,
    LLAMA3: ollamaRequest,
    LLAMA3_3: ollamaRequest,
    LLAMA3_70B: ollamaRequest,
    GEMMA2: ollamaRequest,
    GEMMA2_27B: ollamaRequest,
    BERT_BASE: BERTRequest,
    BERT_LARGE: BERTRequest,
    ALBERT_BASE: BERTRequest,
    ALBERT_LARGE: BERTRequest,
    ROBERTA_BASE: RoBERTaRequest,
    ROBERTA_LARGE: RoBERTaRequest,
    BERTTWEET_BASE: RoBERTaRequest,
    BERTTWEET_LARGE: RoBERTaRequest
}

def generateSentences(modelName, numPrediction):
    model, tokenizer = (initialize_models[modelName](modelName)) if modelName in initialize_models else (None, None)
    
    #Checking if there is an existing file with evaluations
    startingFrom, dicSentences = preExistingFile(modelName, numPrediction)
    templateFile = pd.read_csv(DATA_SOURCE+'template_complete.csv')[startingFrom:]
    df = pd.DataFrame.from_dict(dicSentences)    
    print(f"๏ Generating sentences with {modelName} model...")
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Generating with {modelName} model', unit=' sentences', position=0, leave=True):
        
        prompt = f"Provide only one word to replace the token [MASK] necessary to complete the sentence as output, without repeating the initial part or adding any explanations: {row.loc[TEMPLATE]}"
        response = request_models[modelName](prompt, modelName, model, tokenizer, row.loc[TEMPLATE], numPrediction)
        dicSentences[TYPE].append(row.loc[TYPE])
        dicSentences[TEMPLATE].append(row.loc[TEMPLATE])
        dicSentences[CATEGORY].append(row.loc[CATEGORY])
        dicSentences[GENERATED].append(response)
        df = pd.DataFrame.from_dict(dicSentences)    
        os.makedirs(OUTPUT_SENTENCES, exist_ok=True)
        df.to_csv(f'{OUTPUT_SENTENCES+modelName}_{numPrediction}.csv', index_label = 'index')
    print("๏ File generated!!")

predictionNumber = 1
MODEL_LIST = [GPT4]
for mod in MODEL_LIST:
    generateSentences(mod, predictionNumber)
