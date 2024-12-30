import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#----------------------hyper parameters

d_model = 512
N = 6   
num_heads = 8
d_ff=2048
p_dropout = 0.1
d_k =64
d_v =64
#--------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int, 
                 max_len: int):
        
        super().__init__()
        self.featuredim = dim
        self.maxlen = max_len
        
        PE = torch.zeros(max_len, dim)
        v1 = torch.arange(0,max_len,1)
        v2 = 10000**(-2*torch.arange(0,dim,2)/dim)
        PE[:,::2] = np.sin(torch.outer(v1,v2))
        PE[:,1::2] = np.cos(torch.outer(v1,v2))
        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)
        
    def forward(self, x):
        # shape of x = (B, L, D)     
        _,L,_ = x.shape
        return x + self.PE[:,0:L,:]
    
    
class Embedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int,
                 vocab_size: int):
        super().__init__()
        self.inp_emb= nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(p_dropout)
        
    def forward(self,x):
        return self.dropout(self.pos_enc(self.input_enb(x)))


class Head(nn.Module):
    def __init__(self, d_in:int, d_k:int , d_v:int, mask=None):
        super().__init__()
        
        self.query = nn.Linear(d_in, d_k, bias=None)
        self.key = nn.Linear(d_in,d_k, bias=None)
        self.value = nn.Linear(d_in, d_v, bias=None)
        
        self.d_k = d_k
        
    def forward(self,q,k,v):
        #shape of q,k,v = (B,L,D)
        q=self.query(q)     #(B,L,d_k)
        k=self.key(k)       #(B,L,d_k)
        v=self.value(v)     #(B,L,d_v)
        
        normalization = torch.tensor(torch.sqrt(self.d_k), dtype=torch.float32)
        att_scores = (q @ torch.transpose(k, -2,-1))/normalization
        
        if self.mask is not None:
            att_scores = att_scores.masked_fill(self.mask==0, float('-inf'))
        
        att_scores = F.softmax(att_scores, -1)              #(B,L,L)
        ret = att_scores @ v                                #(B,L,d_v)
        return ret


class MultiHead(nn.Module):
    def __init__(self, num_heads:int, d_in:int, d_k:int, d_v:int, mask=None):
        super().__init__()
        
        self.heads=nn.ModuleList([Head(d_in, d_k, d_v, mask) for _ in range(num_heads)])
        self.p_out=nn.Linear(num_heads*d_v, d_in, bias=False)
        
    def forward(self, q,k,v):
        #shape of q,k,v = (B,L, D=d_in)
        H = torch.cat([h(q,k,v) for h in self.heads],-1) #(B,L,d_v*num_heads)
        return self.p_out(H)
    
    

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mha = MultiHead(num_heads, d_model, d_k, d_v)
        self.dropout1 = nn.Dropout(p_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        
        
        self.fc = nn.Sequential(nn.Linear(d_model,d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff,d_model))
        self.dropout2 = nn.Dropout(p_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # shape of x = (B,L, D=d_model)
        x = self.ln1(self.dropout1(self.mha(x,x,x))+x)
        x = self.ln2(self.dropout2(self.fc(x,x,x))+x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        list_of_blocks = [EncoderBlock() for _ in range(N)]
        
        self.blocks = nn.Sequential(*list_of_blocks)
    
    def forward(self, x):
        #shape of seq = (B, L, 1)
        return self.blocks(x)
  
class DecoderBlock(nn.Module):
    def __init__(self, mask):
        super().__init__()
        
        self.mhawmask = MultiHead(num_heads, d_model, d_k, d_v, mask)
        self.dropout1 = nn.Dropout(p_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.mha = MultiHead(num_heads, d_model, d_k, d_v)
        self.dropout2 = nn.Dropout(p_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.fc = nn.Sequential(nn.Linear(d_model,d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff,d_model))
        self.dropout3 = nn.Dropout(p_dropout)
        self.ln3 = nn.LayerNorm(d_model)
        
    def forward(self, x, y):
        #shape of x,y = (B,L, d_model)
        x = self.ln1(self.dropout1(self.mhawmask(x,x,x))+x)
        x = self.ln2(self.dropout2(self.mha(x,y,y))+x)
        return self.ln3(self.dropout3(self.fc(x))+x)
    
class Decoder(nn.Module):
    def __init__(self, mask_dec):
        super().__init__()
        
        self.blocks = nn.ModuleList([DecoderBlock(mask_dec) for _ in range(N)])
        
    def forward(self, x,y):
        for block in self.blocks:
            x = block(x,y)
        return x
    

class Transformer(nn.Module):
    def __init__(self, vocab_size_src:int, vocab_size_tgt:int , max_len:int, mask=None):
        super().__init__()
        
        self.emb_src = Embedding(d_model, max_len, vocab_size_src)
        self.emb_tgt = Embedding(d_model, max_len, vocab_size_tgt)
        self.encoder = Encoder()
        self.decoder = Decoder(mask)
        self.lin = nn.Linear(d_model, vocab_size_tgt)
    
    def forward(self, src, tgt):
        y = self.emb_src(src)
        y = self.encoder(y)
        
        x = self.emb_tgt(tgt)
        x = self.Decoder(x,y)
        
        x = self.lin(x)
        return F.softmax(x,-1)
        