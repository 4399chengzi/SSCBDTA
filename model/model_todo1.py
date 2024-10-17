"""
MolTrans model - Double Towers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from torch.nn.parameter import Parameter
from _utils.util import fix_random_seed_for_reproduce
from cc_attention.functions import CrissCrossAttention

# Set seed for reproduction
fix_random_seed_for_reproduce(torch_seed=2, np_seed=3)

class MolTransModel(nn.Sequential):
    """
    Interaction Module
    """
    def __init__(self, model_config):
        """
        Initialization
        """
        super(MolTransModel, self).__init__()
        # Basic config
        self.model_config = model_config
        self.drug_max_seq = model_config['drug_max_seq']
        self.target_max_seq = model_config['target_max_seq']
        self.emb_size = model_config['emb_size']
        self.dropout_ratio = model_config['dropout_ratio']
        self.input_drug_dim = model_config['input_drug_dim']
        self.input_target_dim = model_config['input_target_dim']
        
        self.complete_drug_token = model_config['complete_drug_token']
        self.complete_target_token = model_config['complete_target_token']

        self.layer_size = model_config['layer_size']
        self.gpus = torch.cuda.device_count()

        # Model config
        self.interm_size = model_config['interm_size']
        self.num_attention_heads = model_config['num_attention_heads']
        self.attention_dropout_ratio = model_config['attention_dropout_ratio']
        self.hidden_dropout_ratio = model_config['hidden_dropout_ratio']
        self.flatten_dim = model_config['flatten_dim']
        self.hidden_size = model_config['emb_size']

        # Enhanced embeddings
        self.drug_emb = EnhancedEmbedding(self.input_drug_dim, self.emb_size, self.drug_max_seq, self.dropout_ratio)
        self.target_emb = EnhancedEmbedding(self.input_target_dim, self.emb_size, self.target_max_seq, 
                                             self.dropout_ratio)

        # complete position embedding
        self.complete_drug_pos_emb = nn.Sequential(
                nn.Embedding(self.complete_drug_token, self.emb_size),
                PositionalEncoding(self.emb_size)
        )
        self.complete_target_pos_emb = nn.Sequential(
                nn.Embedding(self.complete_target_token, self.emb_size),
                PositionalEncoding(self.emb_size)
        )
        
        # Encoder module
        self.encoder = EncoderModule(self.layer_size, self.hidden_size, self.interm_size, self.num_attention_heads, 
                                      self.attention_dropout_ratio, self.hidden_dropout_ratio)
        # Cross information        
        self.interaction_cnn = nn.Sequential(
                nn.Conv2d(self.hidden_size, 32, 3, padding=1), # TODO
                CrissCrossAttention(32)
        )

        # Decoder module
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),

            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.ReLU(),

            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, d, t, d_masking, t_masking,
                complete_d, complete_t, complete_d_masking, complete_t_masking):
        
        """
        Double Towers
        """
        tempd_masking = d_masking.unsqueeze(1).unsqueeze(2)
        tempt_masking = t_masking.unsqueeze(1).unsqueeze(2)
        
        """
        position embedding
        """
        tempd_masking = (1.0 - tempd_masking) * -10000.0  # Bs, 1,1, len(drug)
        tempt_masking = (1.0 - tempt_masking) * -10000.0  # Bs, 1,1, len(target)

        d_embedding = self.drug_emb(d) # N, len(drug), emb_dim
        t_embedding = self.target_emb(t) # N, len(protein), emb_dim

        """
        complete position embedding
        """
        # complete_d_masking = complete_d_masking.unsqueeze(1).unsqueeze(2)
        # complete_t_masking = complete_t_masking.unsqueeze(1).unsqueeze(2)
        
        complete_d_embedding = self.complete_drug_pos_emb[1](self.complete_drug_pos_emb[0](complete_d) * math.sqrt(self.emb_size))  # Bz, complete_maxlen(drug), emb_size
        complete_t_embedding = self.complete_target_pos_emb[1](self.complete_target_pos_emb[0](complete_t) * math.sqrt(self.emb_size))   # Bz, complete_maxlen(target), emb_size

        complete_d_embedding = torch.mean(complete_d_embedding, dim=1).unsqueeze(1)
        complete_t_embedding = torch.mean(complete_t_embedding, dim=1).unsqueeze(1)
        """
        complete drug/target encoder(transformer) module
        """
        # complete_d_encoder = self.encoder(complete_d_embedding.float(), complete_d_masking)
        # complete_t_encoder = self.encoder(complete_t_embedding.float(), complete_t_masking)
        # complete_d_encoder = torch.mean(complete_d_encoder, dim=1).unsqueeze(1)
        # complete_t_encoder = torch.mean(complete_t_encoder, dim=1).unsqueeze(1)

        """
        transformer module
        """
        d_embedding = d_embedding + complete_d_embedding
        t_embedding = t_embedding + complete_t_embedding

        d_encoder = self.encoder(d_embedding.float(), tempd_masking.float())  # N, len(drug), emb_dim
        t_encoder = self.encoder(t_embedding.float(), tempt_masking.float())  # N, len(protein), emb_dim

        drug_res = torch.unsqueeze(d_encoder, 2).repeat(1, 1, self.target_max_seq, 1) # N, len(drug),len(protein), emb_dim
        target_res = torch.unsqueeze(t_encoder, 1).repeat(1, self.drug_max_seq, 1, 1) # N, len(drug),len(protein), emb_dim
        
        """
        interaction module
        """
        i_score = drug_res * target_res
        i_scoreT = i_score.view(int(i_score.shape[0] / self.gpus), -1, self.drug_max_seq, self.target_max_seq)
        # i_scoreT = torch.sum(i_scoreT, axis=1)
        # i_scoreT = torch.unsqueeze(i_scoreT, 1)
        i_scoreT = F.dropout(i_scoreT, p=self.dropout_ratio) # N, emb_dim, len(drug), len(protein)
        i_scoreT = self.interaction_cnn(i_scoreT).contiguous()
        i_res = i_scoreT.view(int(i_scoreT.shape[0] / self.gpus), -1)
        
        res = self.decoder(i_res)
        return res


class EnhancedEmbedding(nn.Module):
    """
    Enhanced Embeddings of drug, target
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_ratio):
        """
        Initialization
        """
        super(EnhancedEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_id):
        """
        Embeddings
        """
        seq_len = input_id.size(1)
        position_id = torch.arange(0,seq_len, dtype=torch.int64).cuda()
        position_id = position_id.unsqueeze(0).expand_as(input_id)

        word_embeddings = self.word_embedding(input_id)
        position_embeddings = self.position_embedding(position_id)

        embedding = word_embeddings + position_embeddings
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class LayerNorm(nn.Module):
    """
    Customized LayerNorm
    """
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """
        Initialization
        """
        super(LayerNorm, self).__init__()
        # self.beta = paddle.create_parameter(shape=[hidden_size], dtype="float32", 
        #     default_initializer = nn.initializer.Assign(torch.zeros([hidden_size], "float32")))
        # self.gamma = paddle.create_parameter(shape=[hidden_size], dtype="float32", 
        #     default_initializer = nn.initializer.Assign(torch.ones([hidden_size], "float32")))
        self.beta = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.gamma = Parameter(torch.ones(hidden_size, dtype=torch.float32))

        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        """
        LayerNorm
        """
        v = x.mean(-1, keepdim=True)
        s = (x - v).pow(2).mean(-1, keepdim=True)
        x = (x - v) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class EncoderModule(nn.Module):
    """
    Encoder Module with multiple layers
    """
    def __init__(self, layer_size, hidden_size, interm_size, num_attention_heads, 
                 attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(EncoderModule, self).__init__()
        module = Encoder(hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.module = nn.ModuleList([module for _ in range(layer_size)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Multiple encoders
        """
        for layer_module in self.module:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states

  
class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, hidden_size, interm_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio)
        self.latent = LatentModule(hidden_size, interm_size)
        self.output = Output(interm_size, hidden_size, hidden_dropout_ratio)

    def forward(self, hidden_states, attention_mask):
        """
        Encoder block
        """
        attention_temp = self.attention(hidden_states, attention_mask)
        latent_temp = self.latent(attention_temp)
        module_output = self.output(latent_temp, attention_temp)
        return module_output


class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_dropout_ratio)
        self.output = SelfOutput(hidden_size, hidden_dropout_ratio)

    def forward(self, input_tensor, attention_mask):
        """
        Attention block
        """
        attention_output = self.self(input_tensor, attention_mask)
        self_output = self.output(attention_output, input_tensor)
        return self_output


class LatentModule(nn.Module):
    """
    Intermediate Layer
    """
    def __init__(self, hidden_size, interm_size):
        """
        Initialization
        """
        super(LatentModule, self).__init__()
        self.connecter = nn.Linear(hidden_size, interm_size)

    def forward(self, hidden_states):
        """
        Latent block
        """
        hidden_states = self.connecter(hidden_states)
        #hidden_states = F.relu(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class Output(nn.Module):
    """
    Output Layer
    """
    def __init__(self, interm_size, hidden_size, hidden_dropout_ratio):
        """
        Initialization
        """
        super(Output, self).__init__()
        self.connecter = nn.Linear(interm_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """
        Output block
        """
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfOutput(nn.Module):
    """
    Self-Output Layer
    """
    def __init__(self, hidden_size, hidden_dropout_ratio):
        """
        Initialization
        """
        super(SelfOutput, self).__init__()
        self.connecter = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_ratio)

    def forward(self, hidden_states, input_tensor):
        """
        Self-output block
        """
        hidden_states = self.connecter(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfAttention(nn.Module):
    """
    Self-Attention
    """
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_ratio):
        """
        Initialization
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                 "The hidden size (%d) is not a product of the number of attention heads (%d)" % 
                 (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_size

        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_ratio)

    def score_transpose(self, x):
        """
        Score transpose
        """
        # import pdb; pdb.set_trace()
        # temp = x.size()[:-1] + [self.num_attention_heads, self.head_size]
        # x = x.view(*temp)
        x = x.view(x.size(0), x.size(1), self.num_attention_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        Self-Attention block
        """
        temp_q = self.q(hidden_states)
        temp_k = self.k(hidden_states)
        temp_v = self.v(hidden_states)

        q_layer = self.score_transpose(temp_q)
        k_layer = self.score_transpose(temp_k)
        v_layer = self.score_transpose(temp_v)

        attention_score = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.head_size)
        attention_score = attention_score + attention_mask

        attention_prob = nn.Softmax(dim=-1)(attention_score)
        attention_prob = self.dropout(attention_prob)

        attention_layer = torch.matmul(attention_prob, v_layer)
        attention_layer = attention_layer.permute(0, 2, 1, 3).contiguous()

        # import pdb;pdb.set_trace()
        # temp_attention_layer = attention_layer.size()[:-2] + [self.all_head_size]
        attention_layer = attention_layer.view(attention_layer.size(0), attention_layer.size(1), self.all_head_size)
        # attention_layer = attention_layer.view(*temp_attention_layer)
        return attention_layer


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)