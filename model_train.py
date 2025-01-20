import torch 
import torch.nn as nn
import pickle
import json
import pandas as pd
from torch.utils.data import DataLoader
import  torch.nn.functional as F
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from seq2seq_model import Seq2Seq, Seq2SeqAdd
from seq2seq_attn_model import Seq2SeqAttn
from process import get_proc
from hyperParameters import getHyperParameters, getVoacbData


def train_test_split(enc_X, dec_X, split_rate=0.2):
    index = list(range(len(enc_X)))
    random.shuffle(index)
    train_size = int(len(index) * (1 - split_rate))
    train_enc = enc_X[:train_size]
    test_enc = enc_X[train_size:]
    train_dec = dec_X[:train_size]
    test_dec = dec_X[train_size:]
    return train_enc, test_enc, train_dec, test_dec
    pass



class SummaryWrapper:

    def __init__(self):
        self.writer = SummaryWriter("logs")
        self.train_loss_cnt = 0
        self.valid_cnt = 0
    
    def train_loss(self, func):
        def warpper(model, loss_fn, enc_input, dec_input, target, device):
            loss = func(model, loss_fn, enc_input, dec_input, target, device)
            self.writer.add_scalar("train loss", loss, self.train_loss_cnt)
            self.train_loss_cnt += 1
            return loss
        return warpper
    
    def valid_loss_acc(self, func):
        def warpper(model, loss_fn, enc_input, dec_input, target, device):
            loss, accuracy = func(model, loss_fn, enc_input, dec_input, target, device)
            self.writer.add_scalar("valid loss", loss, self.valid_cnt)
            self.writer.add_scalar("valid accuracy", accuracy, self.valid_cnt)
            self.valid_cnt += 1
            return loss, accuracy
        return warpper
sw = SummaryWrapper()


def train(model, loss_fn, optimizer, train_dl, test_dl, device):
    """
    注意: CrossEntropy需要把比较的维度放在第二个维度, 这就是后面交换维度的原因。
        或者,直接reshape(-1,vocab_size)。
    """

    if type(model) == Seq2Seq:
        saved_file = "model_seq2seq.bin"
        print("Seq2Seq")
    elif type(model) == Seq2SeqAdd:
        saved_file = "model_seq2seq_add.bin"
        print("Seq2SeqAdd")
    else:
        saved_file = "model_seq2seq_attn.bin"
        print("Seq2SeqAttn")
    print(device)
    best_loss = torch.inf
    for e in range(epoch):
        model.train()
        bar = tqdm(train_dl)
        update_count = 1
        for enc_input, dec_input, target in bar:
            loss = train_step(model, loss_fn, enc_input, dec_input, target, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_description(f"epoch: {e + 1}, loss: {loss.item():.4f}")

            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), saved_file)
                bar.set_postfix_str(f"Model Saved Successfully!, update_count: {update_count}, loss: {loss.item():.4f}")
                # bar.set_postfix({"loss": loss.item()})
                update_count += 1
    print(f"best_loss: {best_loss:.4f}")

    """
    模型推理时，从因果角度来讲，decoder不应该输入答案来推理。应该输入的是上一次推理的答案。
    """
        # model.eval()
        # with torch.no_grad():
        #     valid_total_loss = 0
        #     for enc_input, dec_input, target in test_dl:
        #         loss, accuracy = valid_step(model, loss_fn, enc_input, dec_input, target, device)
        #         valid_total_loss += loss
        #     print(f"valid_total_loss: {valid_total_loss.item():.4f}, accuracy: {(accuracy * 100):.4f}%")


    # 模型保存 
    # torch.save(model.state_dict(), saved_file)
    # print("Model Saved Successfully!")


@sw.train_loss
def train_step(model, loss_fn, enc_input, dec_input, target, device):
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
        y_hat = model(enc_input, dec_input)  # (batch_size, seq_len, vocab_size)
        y_hat = y_hat.transpose(-1, -2)      # (batch_size, vocab_size, seq_len)
        loss = loss_fn(y_hat, target)        # target: (batch_size, seq_len)
        return loss

@sw.valid_loss_acc
def valid_step(model, loss_fn, enc_input, dec_input, target, device):
        enc_input, dec_input, target = enc_input.to(device), dec_input.to(device), target.to(device)
        y_hat = model(enc_input, dec_input)
        _, _, vocab_size = y_hat.shape
        
        y_hat = y_hat.reshape(-1, vocab_size)
        target = target.reshape(-1)

        loss = loss_fn(y_hat, target)
        y_pred = y_hat.argmax(-1)
        accuracy = (y_pred == target).float().mean()
        return loss, accuracy


if __name__ == "__main__":

    # # 数据和字典准备
    # with open("vocab.bin", "br") as f:
    #    evoc, dvoc =  pickle.load(f)
    # with open("encoder.json", "br") as f:
    #     encoder_data = json.load(f)

    # with open("decoder.json", "br") as f:
    #     decoder_data = json.load(f)

    # batch_size = 64
    # lr = 1e-3
    # embedding_size = 256
    # rnn_hidden_size = 128
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # enc_vocab_size = len(evoc)
    # dec_vocab_size = len(dvoc)  
    # epoch = 20
    
    
    epoch, batch_size, lr, embedding_size, rnn_hidden_size, device = getHyperParameters()
    evoc, dvoc, encoder_data, decoder_data, enc_vocab_size, dec_vocab_size = getVoacbData()

    # 数据读取
    enc_train, enc_test, dec_train, dec_test = train_test_split(encoder_data, decoder_data, split_rate=0.1)

    train_ds = list(zip(enc_train, dec_train))
    test_ds = list(zip(enc_test, dec_test))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=get_proc(evoc, dvoc))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=get_proc(evoc, dvoc))

    # for enc_input, dec_input, target in dataloader:
    #     print(enc_input.shape)
    #     print(enc_input)
    #     print(dec_input.shape)
    #     print(dec_input)
    #     print(target.shape)
    #     print(target)
    #     break
    # pass


    # 模型构建
    model = Seq2Seq(    # 隐变量cat
        enc_vocab_size=enc_vocab_size,
        dec_vocab_size=dec_vocab_size,
        emb_dim=embedding_size,
        rnn_hidden_size=rnn_hidden_size
    )
    model = model.to(device)

        
    model_add = Seq2SeqAdd(     # 隐变量add
        enc_vocab_size=enc_vocab_size,
        dec_vocab_size=dec_vocab_size,
        emb_dim=embedding_size,
        rnn_hidden_size=rnn_hidden_size
    )
    model_add = model_add.to(device)
    
    model_attn = Seq2SeqAttn(     # attn
        enc_vocab_size=enc_vocab_size,
        dec_vocab_size=dec_vocab_size,
        emb_dim=embedding_size,
        rnn_hidden_size=rnn_hidden_size
    )
    model_attn = model_attn.to(device)

    #
    # # 损失函数和优化器
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # # 训练
    # train(model, loss_fn, optimizer, train_dl, test_dl, device)
    

    # # 损失函数和优化器
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model_add.parameters(), lr=lr)
    # # 训练
    # train(model_add, loss_fn, optimizer, train_dl, test_dl, device)


    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_attn.parameters(), lr=lr)
    # 训练
    train(model_attn, loss_fn, optimizer, train_dl, test_dl, device)

