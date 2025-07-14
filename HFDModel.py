import torch
import torch.nn as nn
import math
import transformer


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0: -1]

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # Input: seq_len x batch_size x dim, Output: seq_len, batch_size, dim
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0)
    
    
class HFDModel(nn.Module):
    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout=0.1):
        super(HFDModel, self).__init__()
        
        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )
        
        input_size_2 = int(input_size / 2)
        self.trunk_net_2 = nn.Sequential(
            nn.Linear(input_size_2, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )
        
        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)
        
        self.batch_norm = nn.BatchNorm1d(batch)
        
        self.tar_net = nn.Sequential(
            nn.Linear(emb_size, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, input_size),
        )
        
        self.class_net = nn.Sequential(
            nn.Linear(emb_size, nhid_task),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(p = 0.3),
            nn.Linear(nhid_task, nhid_task),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(p = 0.3),
            nn.Linear(nhid_task, nclasses)
        )
        
    
    def forward(self, x, task_type):        
        if task_type == 'reconstruction':
            x = self.trunk_net(x.permute(1, 0, 2))
            x, attn = self.transformer_encoder(x)
            x = self.batch_norm(x)
        
            output = self.tar_net(x).permute(1, 0, 2)
        elif task_type == 'classification':
            x1 = x[:, :, 0:2]
            x1 = self.trunk_net_2(x1.permute(1, 0, 2))
            x1, attn1 = self.transformer_encoder(x1)
            x1 = self.batch_norm(x1)
            
            x2 = x[:, :, 2:4]
            x2 = self.trunk_net_2(x2.permute(1, 0, 2))
            x2, attn2 = self.transformer_encoder(x2)
            x2 = self.batch_norm(x2)
            
            x = self.trunk_net(x.permute(1, 0, 2))
            x, attn = self.transformer_encoder(x)
            x = self.batch_norm(x)
            
            input_x = x[-1] + x1[-1] + x2[-1]
            
            output = self.class_net(input_x)
        
        return output, attn
