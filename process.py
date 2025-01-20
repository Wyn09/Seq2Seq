import re
import torch 
import pickle
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 读取并拆分数据
def read_data(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        enc_data, dec_data = [], []
        for line in lines:
            if line == "":
                continue
            enc, dec = line.split("\t")
            enc_tokens = re.findall(r"[\w']+", enc)
            dec_tokens = ["<BOS>"] + re.findall(r"[\u4e00-\u9fff]", dec) + ["<EOS>"]
            enc_data.append(enc_tokens)
            dec_data.append(dec_tokens)

    assert len(enc_data) == len(dec_data), "编码数据和解码数据长度不一致。"
    return enc_data, dec_data

# 构建token字典
class Vocabulary:
    
    def __init__(self, vocab):
        self.vocab = vocab

    @classmethod
    def from_documents(cls, documents):
        tokens = set() 
        for cmt in documents:
            tokens.update(list(cmt))
        tokens.discard("<BOS>") # 在dec_data中已经存在"<BOS>", "<EOS>",需要先删除，否则会分别有两个"<BOS>", "<EOS>"
        tokens.discard("<EOS>")
        # set是无序的，可以在list之后做排序,保证每次构建词典顺序一致
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + sorted(list(tokens)) 
        vocab = {token:i for i, token in enumerate(tokens)} 
        return cls(vocab)
        
def get_proc(enc_voc, dev_voc):    

    """
    嵌套函数定义：
        外部函数变量生命周期会延续到内部函数内部调用结束（闭包）

    """
    def batch_fn(data):
        enc_list, dec_list, target_list = [], [], []
        for enc, dec in data:
            enc_index = [enc_voc.get(tk, 1) for tk in enc]
            dec_index = [dev_voc.get(tk, 1) for tk in dec]
            enc_list.append(torch.tensor(enc_index, dtype=torch.long))
            dec_list.append(torch.tensor(dec_index[:-1], dtype=torch.long))
            target_list.append(torch.tensor(dec_index[1:], dtype=torch.long))

        # 用批次中最长的序列长度作为最大长度 ： (batch_size ,max_token_len)
        # 把元素为张量的List转换为张量矩阵，自动填充
        enc_input = pad_sequence(enc_list, batch_first=True)
        dec_input = pad_sequence(dec_list, batch_first=True)
        target = pad_sequence(target_list, batch_first=True)
        return enc_input, dec_input, target
        pass
    return batch_fn


if __name__ == "__main__":
    enc_data, dec_data = read_data("cmn.txt") # 二维列表，存储的是被拆分的每句话
    enc_vocab = Vocabulary.from_documents(enc_data)
    dec_vocab = Vocabulary.from_documents(dec_data)
    print("编码字典长度：", len(enc_vocab.vocab))
    print("解码字典长度：", len(dec_vocab.vocab))
    dataset = list(zip(enc_data, dec_data))
    # callback 回调函数
    dataloader = DataLoader(dataset, batch_size=2,shuffle=True, collate_fn=get_proc(enc_vocab.vocab, dec_vocab.vocab)) 
    # for enc_input, dec_input, target in dataloader:
    #     print(enc_input.shape)
    #     print(enc_input)
    #     print(dec_input.shape)
    #     print(dec_input)
    #     print(target.shape)
    #     print(target)
    #     break
    # pass

    # 数据整体json保存 (json)
    with open("encoder.json", "w", encoding="utf-8") as f:
        json.dump(enc_data, f)

    with open("decoder.json", "w", encoding="utf-8") as f:
        json.dump(dec_data, f)

    with open("vocab.bin", "bw") as f:
        pickle.dump((enc_vocab.vocab, dec_vocab.vocab), f)



    # # 数据每行都是json数据(jsonl)
    # with open("encoders.json", "w", encoding="utf-8") as f:
    #     for enc in enc_data:
    #         str_json = json.dumps(enc)
    #         f.write(str_json + "\n")
































