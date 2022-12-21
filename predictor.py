import torch
from model import TextCNN, Model_MultiTask
from transformers import BertTokenizer

class Predictor():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('./bert')
        self.models = [TextCNN(), Model_MultiTask('label_category'), Model_MultiTask('label_vector')]
        self.params = ['./params/taskA_0.8145.bin', './params/taskB_0.6019.bin', './params/taskC_0.4208.bin']
        self.load_params()
    
    def load_params(self):
        for i in range(3):
            checkpoint = torch.load(self.params[i])
            self.models[i].load_state_dict(checkpoint['model_state_dict'], strict=False)

    def get_batch(self, text):
        tokens = self.tokenizer(text, max_length=250, padding='max_length', truncation=True)
        input_ids = torch.tensor([tokens['input_ids']])
        attention_mask = torch.tensor([tokens['attention_mask']])
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        return batch
    
    def predict(self, text):
        result = []
        id2label = [
            ['not sexist', 'sexist'],
            ['1. threats, plans to harm and incitement', '2. derogation', '3. animosity', '4. prejudiced discussions'],
            ['1.1 threats of harm', '1.2 incitement and encouragement of harm', '2.1 descriptive attacks', '2.2 aggressive and emotive attacks', '2.3 dehumanising attacks & overt sexual objectification', '3.1 casual use of gendered slurs, profanities, and insults', '3.2 immutable gender differences and gender stereotypes', '3.3 backhanded gendered compliments', '3.4 condescending explanations or unwelcome advice', '4.1 supporting mistreatment of individual women', '4.2 supporting systemic discrimination against women as a group']
        ]
        task_C_range = [(0, 2), (2, 5), (5, 9), (9, 11)]
        batch = self.get_batch(text)
        
        for i in range(3):
            self.models[i].eval()
            with torch.no_grad():
                pred_label, logits = self.models[i](batch)
            logits = logits.squeeze()
            label = id2label[i]
            
            if i == 1:
                task_B_pred = pred_label.item()
            elif i == 2:
                logits = logits[task_C_range[task_B_pred][0] : task_C_range[task_B_pred][1]]
                label = id2label[2][task_C_range[task_B_pred][0] : task_C_range[task_B_pred][1]]

            probability = torch.softmax(logits, dim=0)
            result.append([{ 'label': label[i], 'probability': probability[i].item() } for i in range(len(label))])
            
            if i == 0 and pred_label == 0:
                break
        
        return result
