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





class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout ):
        super(EncoderBlock).__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout)] for _ in range(2))
        self.dropout=dropout

    def forward(self,x,src_mask): #src_mask to dont let padidng interact
        x=self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x=self.residual_connections[1](x,lambda x: self.feed_forward_block(x))
        return x
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList):       #we want interable encoder block
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)

        return self.norm(x)
    





class Decoder_Block(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block,feed_forward_block:FeedForwardBlock,dropout):
        self.self_attention_block=self_attention_block
        self.cross_atttention_block=cross_attention_block
        self.feed_forward_block=feed_forward_block
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout)] for _ in range(3))

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connections[2](x,self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class Projection_Layer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.d_model=d_model
        # self.dk=dk
        self.linear1=nn.Linear(d_model,vocab_size)
        # self.linear2=nn.linear(dk,vocab_size)

    def forward(self,x):  #[batch,seq_len,d_model]->[batch,seq_len,vocab_size]
        return torch.log_softmax(self.linear1(x),dim=1)




class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,src_embd:INPUT_EMBEDDING,tgt_embd:INPUT_EMBEDDING,src_pos: Positional_encoding,tgt_pos:Positional_encoding,projection:Projection_Layer):
        self.encoder=encoder
        self.decoder=decoder
        self.src_embd=src_embd
        self.tgt_embd=tgt_embd
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=Projection_Layer
    
    #now we will define 3 methods - one to encode one to decode and one to project and we dont use
    # forward method because during inferincing we can use the output of the encoder and we dont need to ccalculate every time
    def encode(self,src,src_mask):
        src=self.src_embd(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,tgt,encoder_output,src_mask,tgt_mask):
        tgt=self.tgt_embd(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
# The build_transformer function is a factory method that constructs the 
# entire Transformer model architecture. It does not process any data
#  but rather sets up the model's structure, which can then be utilized
#  in the forward method of the Transformer class to process input sequences.

def build_transformer(src_vocab_size,tgt_vocab_size,src_seq_len,tgt_seq_len,d_model=512,N=6,h=8,dropout:0.1,d_ff=2048):
    #create an embedding layer
    src_embed=INPUT_EMBEDDING(d_model,src_vocab_size)
    tgt_embed=INPUT_EMBEDDING(d_model,tgt_vocab_size)

    src_pos=Positional_encoding(d_model,src_seq_len,dropout)
    tgt_pos=Positional_encoding(d_model,tgt_seq_len,dropout)
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_ff,d_model,dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_ff,d_model,dropout)
        decoder_block=Decoder_Block(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    projection_Layer=Projection_Layer(d_model,tgt_vocab_size)
    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_Layer)
    #initialise the parameters
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)

    return transformer