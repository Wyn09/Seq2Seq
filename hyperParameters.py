import pickle
import json
import torch
def getHyperParameters():
    batch_size = 64
    learning_rate = 1e-3
    embedding_size = 256
    rnn_hidden_size = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 100
    return epoch, batch_size, learning_rate, embedding_size, rnn_hidden_size, device

def getVoacbData():
    # 数据和字典准备
    with open("vocab.bin", "br") as f:
       evoc, dvoc =  pickle.load(f)

    with open("encoder.json", "br") as f:
        encoder_data = json.load(f)

    with open("decoder.json", "br") as f:
        decoder_data = json.load(f)

    enc_vocab_size = len(evoc)
    dec_vocab_size = len(dvoc)  
    return evoc, dvoc, encoder_data, decoder_data, enc_vocab_size, dec_vocab_size
        
