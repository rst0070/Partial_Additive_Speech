import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AAMSoftmaxLoss(nn.Module):
    
    def __init__(self, embedding_size:int, num_class:int, scale_constant = 15., margin = 0.3):
        super(AAMSoftmaxLoss, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_class = num_class
        
        self.scale_constant = scale_constant
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.weights = nn.Linear(in_features=embedding_size, out_features=num_class, bias=False)
        
        self.criterian = nn.CrossEntropyLoss()
        
        message = f"AAM Softmaxloss initialized.\nscale: {scale_constant}\nmargin: {margin}"
        print(message)
        
    def forward(self, embeddings, labels):
        
        embeddings = F.normalize(embeddings, dim=1)
        
        for weight_matrix in self.weights.parameters():
            weight_matrix = F.normalize(weight_matrix, dim=1)
        
        # ------- outputs : margined=cos_margined, non-margined=cos_theta ------- #
        cos_theta = self.weights(embeddings)
        sin_theta =  torch.sqrt((1.0 - torch.mul(cos_theta, cos_theta)).clamp(min=1e-40, max=1.))
        
        cos_margined = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        cos_margined = torch.where((cos_theta - self.th) > 0, cos_margined, cos_theta - self.mm)
        
        # ------- gather outputs ------- #
        onehot = torch.zeros_like(cos_theta)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (onehot * cos_margined) + ((1.0-onehot) * cos_theta)
        output = self.scale_constant * output
        
        loss = self.criterian(output, labels)
        if torch.isnan(loss):
            print('nan is occured!')
            print('loss', loss)
            print('output', output)
            print('onehot', onehot)
            print('cos margined', cos_margined)
            print('sin theta', sin_theta)
            print('cos theta', cos_theta)
        return loss