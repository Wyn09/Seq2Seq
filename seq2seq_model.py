import torch 
import torch.nn as nn


            
class Seq2Seq(nn.Module):
    
    def __init__(self, 
                 enc_vocab_size, 
                 dec_vocab_size,
                 emb_dim,
                 rnn_hidden_size):
        
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=enc_vocab_size,
                embedding_dim=emb_dim,
                padding_idx=0
            ),
            nn.LSTM(
                input_size=emb_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) # 双向rnn , output, (h_n, c_n) 其中，h_n和c_n 的shape：(2, batch_size, hidden_size)
              # 要转换为 (1, batch_size, hidden_size * 2)
        )

        self.decoder_emb = nn.Embedding(
                num_embeddings=dec_vocab_size,
                embedding_dim=emb_dim,
                padding_idx=0)
            
        self.dec_rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=rnn_hidden_size * 2,
                num_layers=1,
                batch_first=True, # 只会影响input和output，不影响隐变量
                bidirectional=False)
        self.dec_fc = nn.Linear(rnn_hidden_size * 2, dec_vocab_size)
        pass

    def forward(self, enc_X, dec_X):
        # encoder
        _, (h_n, c_n) = self.encoder(enc_X)
        # 特征值合并，cat, add(decoder的rnn hidden_size就不用*2了)
        h_n = torch.permute(h_n, (1, 0 ,2)).reshape(h_n.shape[1],-1).unsqueeze(0)
        c_n = torch.permute(c_n, (1, 0 ,2)).reshape(c_n.shape[1],-1).unsqueeze(0)
        if self.training:
            # decoder
            dec_emb = self.decoder_emb(dec_X)
            rnn_out, _ = self.dec_rnn(dec_emb, (h_n, c_n)) # encoder状态值 == decoder状态初始值
            # rnn_out.shape: (batch_size, seq_len, hidden_size)
            output = self.dec_fc(rnn_out)
            # output.shape: (batch_size, seq_len, dec_vocab_size)
            return output
            pass


class Seq2SeqAdd(nn.Module):
    """
    特征值合并: add(decoder的rnn hidden_size就不用*2了)
    """
    def __init__(self, 
                 enc_vocab_size, 
                 dec_vocab_size,
                 emb_dim,
                 rnn_hidden_size):
        
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=enc_vocab_size,
                embedding_dim=emb_dim,
                padding_idx=0
            ),
            nn.LSTM(
                input_size=emb_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) # 双向rnn , output, (h_n, c_n) 其中，h_n和c_n 的shape：(2, batch_size, hidden_size)
              # 要转换为 (1, batch_size, hidden_size * 2)
        )

        self.decoder_emb = nn.Embedding(
                num_embeddings=dec_vocab_size,
                embedding_dim=emb_dim,
                padding_idx=0)
            
        self.dec_rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                batch_first=True, # 只会影响input和output，不影响隐变量
                bidirectional=False)
        self.dec_fc = nn.Linear(rnn_hidden_size, dec_vocab_size)
        pass

    def forward(self, enc_X, dec_X):
        # encoder
        _, (h_n, c_n) = self.encoder(enc_X)
        # 特征值合并，cat, add(decoder的rnn hidden_size就不用*2了)
        h_n = (h_n[0] + h_n[1]).unsqueeze(0)
        c_n = (c_n[0] + c_n[1]).unsqueeze(0)
        
        # decoder
        dec_emb = self.decoder_emb(dec_X)
        rnn_out, _ = self.dec_rnn(dec_emb, (h_n, c_n)) # encoder状态值 == decoder状态初始值
        # rnn_out.shape: (batch_size, seq_len, hidden_size)
        output = self.dec_fc(rnn_out)
        # output.shape: (batch_size, seq_len, dec_vocab_size)
        return output
        pass

if __name__ == "__main__":

    model = Seq2Seq(
        enc_vocab_size=100,
        dec_vocab_size=200,
        emb_dim=300,
        rnn_hidden_size=400,
    )
    # print(model)
    enc_input = torch.randint(0,100,(3,10))
    dec_input = torch.randint(0,200,(3,15))
    y_hat = model(enc_input, dec_input)
    print(y_hat.shape)

    
    
    model_add = Seq2SeqAdd(
        enc_vocab_size=100,
        dec_vocab_size=200,
        emb_dim=300,
        rnn_hidden_size=400,
    )
    # print(model_add)
    enc_input = torch.randint(0,100,(3,10))
    dec_input = torch.randint(0,200,(3,15))
    y_hat = model_add(enc_input, dec_input)
    print(y_hat.shape)
    pass