import torch.nn as nn
import torch
import copy
import torch.nn.functional as F

# initialize gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):

    def __init__(self, encoder, d_model=768, nhead=12, num_decoder_layers=8, 
                 dim_feedforward=2048, dropout=0.1, activation="gelu", 
                 trg_vocab_size=5000, max_len= 197):
        super().__init__()

        # initialize encoder
        self.encoder = encoder.to(device)

        # initialize decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # dimension of feature representation vector
        self.d_model = d_model
        # number of heads in the decoder
        self.nhead = nhead

        #initialize unique vector for each word from our dictionary
        self.word_embed = nn.Embedding(trg_vocab_size,d_model)
        # positional embedding is necessary to 
        # remember how words (lexemes) are related to each other
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_trg_mask(self, trg):
        # Create a mask for the target text
        N, trg_len = trg.shape
        trg_mask = torch.triu(torch.full((trg_len, trg_len), float('-inf')), diagonal=1).expand(
            N, 12, trg_len, trg_len
        )

        return trg_mask

    def forward(self, inp_img, inp_text):

        # Pass the input image through encoder
        VIT_output = self.encoder(inp_img, output_hidden_states=True,output_attentions =True)
        # Save predictions
        logits = VIT_output.logits
        class_idx = logits.argmax(-1)
        class_idx = [self.encoder.config.id2label[idx.item()] for idx in class_idx]

        # Target text representation
        N, seq_length = inp_text.shape
        positions = torch.arange(seq_length).expand(N, seq_length)
        enc_text =  self.word_embed(inp_text.to(device))
        enc_text += self.pos_embed(positions.to(device))
        # Mask for target text
        trg_mask = self.make_trg_mask(inp_text).to(device)

        # Text recognition
        dec_text = self.decoder(enc_text, VIT_output.hidden_states[-1], VIT_output.attentions[-1], trg_mask)[0]
        dec_text = self.fc_out(dec_text)

        return class_idx, dec_text

    def generate(self, inp_img, max_new_tokens=197):
        # Pass the test image through encoder
        VIT_output = self.encoder(inp_img, output_hidden_states=True,output_attentions =True)
        # Get and save the class of an image
        logits = VIT_output.logits
        class_idx = logits.argmax(-1)
        class_idx = self.encoder.config.id2label[class_idx.item()]
        # Create empty sentences filled with 0 (padding)
        idx = torch.zeros(max_new_tokens).type(torch.LongTensor).unsqueeze(0)
        N, seq_length = idx.shape
        # Sequential word generation in a sentence
        for k in range(max_new_tokens):
            enc_idx =  self.word_embed(idx.to(device)) + self.pos_embed(torch.arange(seq_length).
                                                                            expand(N, seq_length).to(device))

            trg_mask = self.make_trg_mask(idx).to(device)
            dec_paths = self.decoder(enc_idx, VIT_output.hidden_states[-1], 
                                        VIT_output.attentions[-1], trg_mask)[0]
            dec_paths = self.fc_out(dec_paths)          
            # focus only on the last time step
            logits = dec_paths[:, -1, :] 
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # if token = the end of sentence
            if idx_next==3:
                break
            # append sampled index to the running sequence
            idx[:,k] = idx_next

        return idx

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, 
                tgt, 
                memory,
                memory_pos,
                tgt_pos):
        
        output = tgt


        for layer in self.layers:
            output = layer(output, memory,
                           memory_pos, tgt_pos)
            
        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout,
                 activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.nhead = nhead

    def forward(self, tgt, 
                     memory,
                     memory_pos=None,
                     tgt_pos=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_pos.reshape(self.nhead*tgt.shape[0],
                                                                       tgt.shape[1],tgt.shape[1]))[0]
    
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(query=tgt,
                                   key=memory,
                                   value=memory,
                                   attn_mask=memory_pos.reshape(self.nhead*tgt.shape[0],
                                                                tgt.shape[1],tgt.shape[1]))[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
