from constants import *

def getModelTokenizer(modelName):
    if((modelName == BERT_BASE) or (modelName == BERT_LARGE)):
        return BertForMaskedLM.from_pretrained(MODELS[modelName]), BertTokenizer.from_pretrained(MODELS[modelName])
    elif((modelName == ROBERTA_BASE) or (modelName == ROBERTA_LARGE)):
        return RobertaForMaskedLM.from_pretrained(MODELS[modelName]), RobertaTokenizer.from_pretrained(MODELS[modelName])
    elif(modelName == ALBERT_BASE) or (modelName == ALBERT_LARGE):
        return AlbertForMaskedLM.from_pretrained(MODELS[modelName]), AlbertTokenizer.from_pretrained(MODELS[modelName])
    elif((modelName == BERTTWEET_BASE) or (modelName == BERTTWEET_LARGE)):
        return AutoModel.from_pretrained(MODELS[modelName]), AutoTokenizer.from_pretrained(MODELS[modelName], use_fast=False)
  
def getWordPrediction(modelName, model, tokenizer, numPrediction, text):
    if ((modelName == BERT_BASE) or (modelName == BERT_LARGE) or (modelName == ALBERT_BASE) or (modelName == ALBERT_LARGE)):
        text = "[CLS] %s [SEP]"%text
        tokenized_text = tokenizer.tokenize(text)
        masked_index = tokenized_text.index(MASKBERT)
    else:
        text = text.replace(MASKBERT, MASKROBERT)
        text = "<s> %s </s>"%text
        tokenized_text = tokenizer.tokenize(text)
        masked_index = tokenized_text.index(MASKROBERT)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        output = model(tokens_tensor)
        predictions = output[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, numPrediction, sorted=True)

    adjectiveList = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        if ((modelName == ALBERT_BASE) or (modelName == ALBERT_LARGE)):
            predicted_token = predicted_token.replace(r'▁', '')
        elif ((modelName == ROBERTA_BASE) or (modelName == ROBERTA_LARGE) or (modelName == BERTTWEET_BASE) or (modelName == BERTTWEET_LARGE)):
            predicted_token = predicted_token.replace('Ġ', '')
        adjectiveList.append(predicted_token)
    return adjectiveList  
    
def getPredictions(modelName, numPrediction):
    templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH, sep=";")
    model, tokenizer = getModelTokenizer(modelName)
    prediction = []
    for index,row in tqdm(templateFile.iterrows(), total=templateFile.shape[0], desc=f'Predicting mask with {modelName} in top-{numPrediction}', unit='sentences'):
        model_prediction = getWordPrediction(modelName, model, tokenizer, numPrediction, row.loc['template'])
        prediction.append(model_prediction)
        
    
    templateFile.loc[:,'prediction'] = prediction
    os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
    templateFile.to_csv(f'{OUTPUT_PREDICTION}/{modelName}_{numPrediction}.csv', sep=';', index=False)

#Input: model, number of predictions
for i in range(len(MODELS)):
    modelName = list(MODELS.keys())[i]
    predictionNumber = 1
    getPredictions(modelName, predictionNumber)

