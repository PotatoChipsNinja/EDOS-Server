import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CNNExtractor(nn.Module):
    def __init__(self, feature_kernel, input_dim):
        super(CNNExtractor, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, feature_num, kernel_size) for kernel_size, feature_num in feature_kernel.items()])

    def forward(self, input):
        input = input.permute(0, 2, 1)
        feature = [conv(input) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]).squeeze(dim=-1) for f in feature]
        feature = torch.cat(feature, dim=1)
        return feature

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MLP, self).__init__()
        layers = list()
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.bert=BertModel.from_pretrained('./bert')
        feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = CNNExtractor(feature_kernel, 768)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        num_labels=2
        self.mlp = MLP(mlp_input_shape, [512], num_labels, 0)

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        feature=self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        output = self.convs(feature)
        output = self.mlp(output)
        logits = F.log_softmax(output,1)
        pred_label=torch.argmax(logits, dim=1)
        return pred_label,logits      

class Model_MultiTask(nn.Module):
    def __init__(self, labeltype):
        super(Model_MultiTask, self).__init__()
        self.model = BertModel.from_pretrained('./bert')
        feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = CNNExtractor(feature_kernel, 768)
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        
        self.mlpA=MLP(mlp_input_shape, [512], 2, 0)
        self.mlpB=MLP(mlp_input_shape, [512], 4, 0)
        self.mlpC=MLP(mlp_input_shape, [512], 11, 0)
        # self.softmax=F.log_softmax(dim=1)
        self.task=labeltype

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        bert_out=self.model(input_ids, attention_mask=attention_mask)
        
        feature = self.convs(bert_out.last_hidden_state)

        fea_B=self.mlpB(feature)
        logits_B=F.log_softmax (fea_B,1)
        
        fea_C=self.mlpC(feature)
        logits_C=F.log_softmax(fea_C,1)

        if self.task=='label_category':
            pred_label=torch.argmax(logits_B, dim=1)
            logits=logits_B
        else:
            pred_label=torch.argmax(logits_C, dim=1)
            logits=logits_C
        return pred_label, logits