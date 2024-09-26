import torch
import torch.nn as nn
import math
import torch.nn as nn

class INPUT_EMBEDDING(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class Positional_encoding(nn.Module):
    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        # create a matrix of shape seq_len*d_model
        pe=torch.zeros(seq_len,d_model)
        
        #create a vector for position in seq_len*1
        position=torch.arrange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float())*(-math.log(10000.0)/d_model)         #2 is the step like 2,4,6,8
        #apply sin to even position

        pe[:,0::2]=torch.sin(position/div_term)

        # apply cos to odd position
        pe[:,1::2]=torch.cos(position/div_term)

        pe.unsqueeze(0)       # tensor of shape 1*seq_len*d_model
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1],:]).requires_grad(False)                        #upto x.shape[1] -seq_len add this
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))  #multiplied
        self.bias=nn.Parameter(torch.zeros(1))  #added


    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self,d_ff,d_model,dropout):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):                #x-batch,seq_len,dmodel
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

        



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model,h,dropout):
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h!=0,'d_model not divisible by h'

        self.d_k=d_model//h
        self.dropout=nn.Dropout(dropout)
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)           #transpose_last_two 
        if mask is not None:
            attention_scores.masked_fill(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores*value),attention_scores



    def forward(self,q,k,v,mask):
        query=self.w_q(q)                 #batch,seq_len,d_model-> batch-seq_len,d_model
        key=self.w_k(k)
        value=self.w_v(v)
        #[batch,seq_len,d_model]-> [batch,seq_len,h,dk]-> transpose [batch,head,seq_len,dk]
        query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)
        key.view(query.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)
        value.view(query.shape[0],value.shape[1],self.h,self.dk).transpose(1,2)
        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        #x shape -[batch,head,seq_len,dk] -[batch,seq_len,head,dk] -[batch,seq_len,head*dk] 
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)         
        return self.w_o(x) 
    

class ResidualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()
    def forward(self,x,sublayer):
        return self.dropout(sublayer(self.norm(x)))






