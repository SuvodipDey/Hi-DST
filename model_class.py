import torch
import torch.nn as nn
from transformers import BertModel

# switch_model
class SwitchModel(nn.Module):
    def __init__(self, num_labels):
        super(SwitchModel, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.drop_out = nn.Dropout(0.3)
        self.l1 = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_masks):
        outputs = self.encode(input_ids, attention_masks)
        input1 = torch.mean(outputs[2][-2], dim=1)
        input1 = self.drop_out(input1)
        output1 = self.l1(input1)
        return output1

# domain_model
class DomainModel(nn.Module):
    def __init__(self, domain_matrix, num_labels):
        super(DomainModel, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.embedding_domain = nn.Embedding.from_pretrained(torch.FloatTensor(domain_matrix))
        self.drop_out = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        self.l1 = nn.Linear(300, 768)
        self.l2 = nn.Linear(768*2, num_labels)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_masks, domain_ids):
        outputs = self.encode(input_ids, attention_masks)
        with torch.no_grad():
            input2 = self.embedding_domain(domain_ids)

        input1 = outputs[2][-2]
        input2 = self.l1(input2)
        input2 = self.gelu(input2)
        
        input3=torch.unsqueeze(input2, -1)  
        a = torch.matmul(input1, input3)/28.0
        a = self.smax(torch.squeeze(a, -1))
        a = torch.unsqueeze(a, -1)
        input1 = input1.permute(0, 2, 1)
        input1 = torch.matmul(input1, a)
        input1 = torch.squeeze(input1,-1)
        
        output = torch.cat((input1, input2), 1)
        output = self.drop_out(output)
        output = self.l2(output)
        return output

# slot action model
class SlotActionModel(nn.Module):
    def __init__(self, weights_matrix, domain_matrix, num_labels):
        super(SlotActionModel, self).__init__()
        self.encode = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix))
        self.embedding_domain = nn.Embedding.from_pretrained(torch.FloatTensor(domain_matrix))
        self.drop_out = nn.Dropout(0.3)
        self.gelu = nn.GELU()
        self.l1 = nn.Linear(300*2, 768)
        self.l2 = nn.Linear(768*2, num_labels)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_masks, slot_ids, domain_ids):
        outputs = self.encode(input_ids, attention_masks)
        with torch.no_grad():
            slot_embeddings = self.embedding(slot_ids)
            domain_embeddings = self.embedding_domain(domain_ids)
            input2 = torch.cat((slot_embeddings, domain_embeddings), 1)

        input1 = outputs[2][-2]
        input2 = self.l1(input2)
        input2 = self.gelu(input2)
        input3=torch.unsqueeze(input2, -1)
        
        a = torch.matmul(input1, input3)/28.0
        a = self.smax(torch.squeeze(a, -1))
        a = torch.unsqueeze(a, -1)
        input1 = input1.permute(0, 2, 1)
        input1 = torch.matmul(input1, a)
        input1 = torch.squeeze(input1,-1)
        
        output = torch.cat((input1, input2), 1)
        output = self.drop_out(output)
        output = self.l2(output)
        return output