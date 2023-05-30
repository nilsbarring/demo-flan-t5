import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Handler(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained('./models').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('./models')

    def preprocess(self, data):
        input_text = data
        if input_text is None:
            raise ValueError("no input found")
        inputs = self.tokenizer(input_text, return_tensors="pt")
        return inputs.to(self.device)

    def inference(self, inputs):
        prediction = self.model.generate(**inputs, max_new_tokens=1000)
        return prediction

    def postprocess(self, prediction):
        output = self.tokenizer.decode(prediction[0], skip_special_tokens=True)
        return output
    
    