import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from util import clones
from transformers.activations import get_activation




def self_attention(query, key, value, mask=None):
   key_transpose = torch.transpose(key, -2, -1) # key vector transpose (to perform matrix multiplication)
   matmul_result = torch.matmul(query, key_transpose) # query * key
   d_k = query.size()[-1] # dimension of key vector
   attention_score = matmul_result/math.sqrt(d_k) # calculate attention score
 
   if mask is not None: # 각 마스크의 종류에 따라 어텐션 달라짐 - 2가지 존재: 1. 인코더: 패딩마스크, 2. 디코더: 룩어헤드 마스크, 3. 디코더: 패딩마스크
       attention_score = attention_score.masked_fill(mask == 0, -1e20) # masked position's attention score setting
 
   softmax_attention_score = F.softmax(attention_score, dim=-1)
   result = torch.matmul(softmax_attention_score, value) # the final attention score
 
   return result, softmax_attention_score




class MultiHeadAttention(nn.Module):
   # in the paper, the number of header is 8.
   def __init__(self, head_num=8, d_model = 512, dropout = 0.1):
       super(MultiHeadAttention, self).__init__()
       self.head_num = head_num
       self.d_model = d_model # dimensionality of the model
       self.d_k = self.d_v = d_model // head_num
     
       self.w_q = nn.Linear(d_model, d_model) # query tensor
       self.w_k = nn.Linear(d_model, d_model) # key tensor
       self.w_v = nn.Linear(d_model, d_model) # value tensor
       self.w_o = nn.Linear(d_model, d_model) # output tensor
     
       self.self_attention = self_attention
       self.dropout = nn.Dropout(p=dropout) # dropout rate to apply after the attention calculation(regularization)
 
   def forward(self, query, key, value, mask = None): # forward pass of the MultiHeadAttention
       if mask is not None:
           # same mask applied to all h heads
           mask = mask.unsqueeze(1) # match the batch size of the input tensor
     
       batch_num = query.size(0)
     
       # reshape and transform the tensors to have dimensions[batch_num, sequence_length, head_num, d_k]
       query = self.w_q(query).view(batch_num,-1, self.head_num, self.d_k).transpose(1, 2)
       key = self.w_k(key).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)
       value = self.w_v(value).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)
     
       attention_result, attetion_score = self.self_attention(query, key, value, mask)
       attention_result = attention_result.transpose(1, 2).contiguous().view(batch_num, -1, self.head_num * self.d_k)
       return self.w_o(attention_result)
 
class FeedForward(nn.Module):
   def __init__(self, d_model, dropout=0.1):
       super(FeedForward, self).__init__()
       self.w_1 = nn.Linear(d_model, d_model * 4) # feedforward first weight # 원래 논문에서 2048
       self.w_2 = nn.Linear(d_model * 4, d_model) # second weight
       self.dropout = nn.Dropout(p=dropout)
     
   def forward(self, x): # forward pass of the feedforward
       return self.w_2(self.dropout(F.relu(self.w_1(x)))) # x -> w_1 -> relu -> dropout -> w_2
 
class LayerNorm(nn.Module): # 층 정규화
   # features: number of features
   # eps: small value
 
   def __init__(self, features, eps=1e-6):
       super(LayerNorm, self).__init__()
       self.a_2 = nn.Parameter(torch.ones(features)) # learnable parameters
       self.b_2 = nn.Parameter(torch.zeros(features))
       self.eps = eps
 
   def forward(self,x):
       mean = x.mean(-1, keepdim = True)
       std = x.std(-1, keepdim = True)
     
       return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # layer normalization의 공식




# 잔차 연결
class ResidualConnection(nn.Module):
   def __init__(self, size, dropout):
       super(ResidualConnection, self).__init__()
       self.norm = LayerNorm(size)
       self.dropout = nn.Dropout(dropout)
     
   def forward(self, x, sublayer):
       return x + self.dropout((sublayer(self.norm(x)))) # 정규화한 거 -> sub layer + 입력값




class Encoder(nn.Module):
   def __init__(self, d_model, head_num, dropout):
       super(Encoder, self).__init__()
       self.multi_head_attention = MultiHeadAttention(d_model = d_model, head_num = head_num)
       self.residual_1 = ResidualConnection(d_model, dropout=dropout)
     
       self.feed_forward = FeedForward(d_model)
       self.residual_2 = ResidualConnection(d_model, dropout = dropout)
     
   def forward(self, input, mask): # 트랜스포머 인코더의 구조대로 진행
       x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
       x = self.residual_2(x, lambda x: self.feed_forward(x))
       return x
     
class Decoder(nn.Module):
   def __init__(self, d_model, head_num, dropout):
       super(Decoder, self).__init__()
       self.masked_multi_head_attention = MultiHeadAttention(d_model = d_model, head_num=head_num) # 첫번째 서브층: Masked multi-head attention
       self.residual_1 = ResidualConnection(d_model, dropout=dropout)
     
       self.encoder_decoder_attention = MultiHeadAttention(d_model = d_model, head_num = head_num) # 두번째 서브층: Encoder-Decoder attention
       self.residual_2 = ResidualConnection(d_model, dropout=dropout)
     
       self.feed_forward = FeedForward(d_model)
       self.residual_3 = ResidualConnection(d_model, dropout = dropout)
 
   def forward(self, target, encoder_output, target_mask, encdoer_mask):
       x  = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
       x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encdoer_mask))
       x = self.residual_3(x, self.feed_forward)




       return x
 
class Embeddings(nn.Module):
   # vocab_num: the number of unique words
   def __init__(self, vocab_num, d_model):
       super(Embeddings, self).__init__()
       self.emb = nn.Embedding(vocab_num, d_model)
       self.d_model = d_model
     
   def forward(self, x): # input tensor x: index of the words(tokens)
       return self.emb(x) * math.sqrt(self.d_model) # lookup embedding matrix and returns the corresponding embedding vectors
                                                    # multiplied by sqrt(self.d_model) : scaling factor -> prevent the embedding value become too large / small
 
class PositionalEncoding(nn.Module):
   # max_seq_len: maximum sequence length of the input
   def __init__(self, max_seq_len, d_model, dropout=0.1):
       super(PositionalEncoding, self).__init__()
       self.dropout = nn.Dropout(p=dropout)
     
       pe = torch.zeros(max_seq_len, d_model) # tensor will hold positional encodings
     
       position = torch.arange(0, max_seq_len).unsqueeze(1) # position index tensor: represent the positions of the elements in the input sequence
       base = torch.ones(d_model // 2).fill_(max_seq_len)
       pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model, dtype=torch.float32)
       # pow_term: representing the exponent term in the positional encoding formula
       div_term = torch.pow(base, pow_term)
     
       pe[:, 0::2] = torch.sin(position / div_term) # calculate positional encoding
       pe[:, 1::2] = torch.cos(position / div_term)
     
       pe = pe.unsqueeze(0)
     
       self.register_buffer("positional_encoding", pe) # 학습되지 않는 변수
     
   def forward(self, x): # forward pass of the positional encoding
       x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False) # sliced to match the positional information to the input sequence
       return self.dropout(x)




class Generator(nn.Module): # generate the output logits(predicted probabilities for each token)
   def __init__(self, d_model, vocab_num):
       super(Generator, self).__init__()
       self.proj_1 = nn.Linear(d_model, d_model * 4)
       self.proj_2 = nn.Linear(d_model * 4, vocab_num)
 
   def forward(self, x):
       x = self.proj_1(x)
       x = self.proj_2(x)
       return x




class Transformer(nn.Module):
   def __init__(self, vocab_num, d_model, max_seq_len, head_num, dropout, N):
       super(Transformer, self).__init__()
       self.embedding = Embeddings(vocab_num, d_model) # input embedding
       self.positional_encoding = PositionalEncoding(max_seq_len, d_model) # add positionanl encoding
     
       self.encoders = clones(Encoder(d_model = d_model, head_num=head_num, dropout=dropout), N) # add encoders and decoders
       self.decoders = clones(Decoder(d_model=d_model, head_num=head_num, dropout=dropout), N)
     
       self.generator = Generator(d_model, vocab_num) # add generator to make output logit
     
       # transformer forward function
   def forward(self, input, target, input_mask, target_mask, labels=None):
       x = self.positional_encoding(self.embedding(input)) # embedding -> positional_encoding
     
       for encoder in self.encoders: # Encoder
           x = encoder(x, input_mask)
     
       target = self.positional_encoding(self.embedding(target)) # embedding -> positional_encoding
     
       for decoder in self.decoders: # Decoder
           target = decoder(target, x, target_mask, input_mask)
     
       lm_logits = self.generator(target)
       loss = None
     
       if labels is not None:
           shift_logits = lm_logits[..., :-1, :].contiguous()
           shift_labels = labels[..., 1:].contiguous()
         
           loss_fct = CrossEntropyLoss(ignore_index=0)
           loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
     
       return lm_logits, loss
 
   def encode(self, input, input_mask):
       x = self.positional_encoding(self.embedding(input))
       for encoder in self.encoders:
           x = encoder(x, input_mask)
       return x




   def decode(self, encode_output, encoder_mask, target, target_mask):
       target = self.positional_encoding(self.embedding(target))
       for decoder in self.decoders:
           target = decoder(target, encode_output, target_mask, encoder_mask)
     
       lm_logits = self.generator(target)
       return lm_logits

