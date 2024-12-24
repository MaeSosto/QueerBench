# # Imports
from utilities import *

# # Build Complete Template
class CompleteTemplateBuilder():
    def __init__(self):
        self.template = pd.read_csv(TEMPLATE_PATH, sep=";", dtype={'template': 'category', 'category': 'category'})
        self.nouns = pd.read_csv(NOUNS_PATH, sep=';')
        self.pronouns = pd.read_csv(PRONOUNS_PATH, sep=';')
        self.template_builder()

    def plural_form(self, be, sentence):
        if be == 'are':
            word = sentence.split(" ")[1]
            if word[-1] == 's':
                sentence = sentence.replace(word, word[:-1])
        return sentence

    def template_builder(self):
        dataList =[]
        for index,row in tqdm(self.template.iterrows(), total=self.template.shape[0], desc='Creating template', unit=' sentences'):
            sentence = row.loc['template']
            
            #Creating sentences with nouns
            for ind, r in self.nouns.iterrows():
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
            for ind, r in self.pronouns.iterrows():
                _sentence= self.plural_form(r.loc[BE_], sentence.replace(TARGET_, r.loc[SUBJECT]))
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


# # Generate Predictions
class TemplatePrediction:
    def __init__(self, modelName, numPrediction):
        self.numPrediction = numPrediction
        self.modelName = modelName
        self.templateFile = pd.read_csv(TEMPLATES_COMPLETE_PATH, sep=";")
        self.model, self.tokenizer = self.getModelTokenizer()
        self.getPredictions()

    def getModelTokenizer(self):
        if((self.modelName == BERT_BASE) or (self.modelName == BERT_LARGE)):
            return BertForMaskedLM.from_pretrained(MODELS[self.modelName]), BertTokenizer.from_pretrained(MODELS[self.modelName])
        elif((self.modelName == ROBERTA_BASE) or (self.modelName == ROBERTA_LARGE)):
            return RobertaForMaskedLM.from_pretrained(MODELS[self.modelName]), RobertaTokenizer.from_pretrained(MODELS[self.modelName])
        elif(self.modelName == ALBERT_BASE) or (self.modelName == ALBERT_LARGE):
            return AlbertForMaskedLM.from_pretrained(MODELS[self.modelName]), AlbertTokenizer.from_pretrained(MODELS[self.modelName])
        elif((self.modelName == BERTTWEET_BASE) or (self.modelName == BERTTWEET_LARGE)):
            return AutoModel.from_pretrained(MODELS[self.modelName]), AutoTokenizer.from_pretrained(MODELS[self.modelName], use_fast=False)
    

    def getWordPrediction(self, text):
        if ((self.modelName == BERT_BASE) or (self.modelName == BERT_LARGE) or (self.modelName == ALBERT_BASE) or (self.modelName == ALBERT_LARGE)):
            text = "[CLS] %s [SEP]"%text
            tokenized_text = self.tokenizer.tokenize(text)
            masked_index = tokenized_text.index(MASKBERT)
        else:
            text = text.replace(MASKBERT, MASKROBERT)
            text = "<s> %s </s>"%text
            tokenized_text = self.tokenizer.tokenize(text)
            masked_index = tokenized_text.index(MASKROBERT)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            output = self.model(tokens_tensor)
            predictions = output[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, self.numPrediction, sorted=True)

        adjectiveList = []
        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]
            if ((self.modelName == ALBERT_BASE) or (self.modelName == ALBERT_LARGE)):
                predicted_token = predicted_token.replace(r'▁', '')
            elif ((self.modelName == ROBERTA_BASE) or (self.modelName == ROBERTA_LARGE) or (self.modelName == BERTTWEET_BASE) or (self.modelName == BERTTWEET_LARGE)):
                predicted_token = predicted_token.replace('Ġ', '')
            adjectiveList.append(predicted_token)
        return adjectiveList

    def getPredictions(self):
        prediction = []
        for index,row in tqdm(self.templateFile.iterrows(), total=self.templateFile.shape[0], desc=f'Predicting mask with {self.modelName} in top-{self.numPrediction}', unit='sentences'):
            model_prediction = self.getWordPrediction(row.loc['template'])
            prediction.append(model_prediction)
        self.templateFile.loc[:,'prediction'] = prediction
        os.makedirs(OUTPUT_PREDICTION, exist_ok=True)
        self.templateFile.to_csv(f'{OUTPUT_PREDICTION}/{self.modelName}_{self.numPrediction}.csv', sep=';', index=False)


# # Main
#Create the complete template
#CompleteTemplateBuilder()

#Input: model, number of predictions
for i in range(len(MODELS)):
    modelName = list(MODELS.keys())[i]
    predictionNumber = 5
    TemplatePrediction(modelName, predictionNumber)

