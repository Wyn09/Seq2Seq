import torch 
import torch.nn as nn
import pickle
from seq2seq_attn_model import Seq2SeqAttn
import re
import torch.nn.functional as F
from hyperParameters import getHyperParameters, getVoacbData
"""
# 解码推理流程

## 1. 加载训练好的模型和词典
- 加载已经训练好的模型和词汇表，用于后续的解码推理。

## 2. 解码推理流程
1. 用户输入通过词汇表转换为 token_index:
   - 将用户输入通过词汇表转换为对应的 token 索引。

2. token_index 通过编码器获取 h_n 和 c_n:
   - 使用编码器处理 token_index, 获取隐藏状态 h_n 和细胞状态 c_n。

3. 准备解码器输入的第一个 token_index:
   - 初始化解码器的输入为起始标记 [['B0S']]，其形状为 [1, 1]。

4. 循环解码器:
   - 解码器输入: [['B0S']], h_n, c_n。
   - 解码器输出: output, (h_n, c_n), output 的形状为 [1, 1, dec_voc_size]。
   - 计算 argmax: 从 output 中获取概率最高的 token_index。
   - 解码器的下一个输入: 将得到的 token_index 作为下一个输入。
   - 收集每次的 token_index: 将每次解码得到的 token_index 收集到解码集合中。

5. 输出解码结果:
   - 最终输出解码集合中的 token_index 序列作为解码结果。
"""


def predict(model, enc_input, evoc, dvoc, dvoc_inv, max_dec_len=50):
   model.eval()
   with torch.no_grad():

      enc_input = re.findall(r"[\w(')]+", enc_input)
      enc_idx = torch.tensor([[evoc.get(tk, 1) for tk in enc_input]], dtype=torch.long)

      # 编码器
      enc_output, (h_n, c_n) = model.encoder(enc_idx)
      h_n = torch.permute(h_n, (1, 0 ,2)).reshape(h_n.shape[1],-1).unsqueeze(0)
      c_n = torch.permute(c_n, (1, 0 ,2)).reshape(c_n.shape[1],-1).unsqueeze(0)
      # 解码器 输入shape:(1,1)
      dec_input = torch.tensor([[dvoc["<BOS>"]]])

      dec_tokens = []
      while True:
         output, (h_n, c_n) = model.decoder(dec_input, enc_output, (h_n, c_n))
         # 得到概率最大的token索引
         tk_index = output.argmax(-1)
         # 可以使用采样取值来进行随机输出
         # tk_index = F.softmax(output[-1], -1).multinomial(1)
         dec_input = tk_index
         # 反向字典，查询预测的token
         tk = dvoc_inv[tk_index.item()]
         # 打印预测的 token
         if tk != "<EOS>" and len(dec_tokens) <= max_dec_len:
            dec_tokens.append(tk)
         else:
            break
      return dec_tokens

if __name__ == "__main__":
    
   # 加载模型
   state_dict = torch.load("model_seq2seq_attn.bin", weights_only=False)

   # 加载字典
   # with open("vocab.bin", "rb") as f:
   #    evoc, dvoc = pickle.load(f)
   
   # embedding_size = 256
   # rnn_hidden_size = 128
   # device = "cuda" if torch.cuda.is_available() else "cpu"
   # enc_vocab_size = len(evoc)
   # dec_vocab_size = len(dvoc)  

   epoch, batch_size, lr, embedding_size, rnn_hidden_size, device = getHyperParameters()
   evoc, dvoc, encoder_data, decoder_data, enc_vocab_size, dec_vocab_size = getVoacbData()

   # 创建解码器
   model = Seq2SeqAttn(
      enc_vocab_size=enc_vocab_size,
      dec_vocab_size=dec_vocab_size,
      emb_dim=embedding_size,
      rnn_hidden_size=rnn_hidden_size
   )
   model.load_state_dict(state_dict)
   # 构建反向字典
   dvoc_inv = {v: k for k ,v in dvoc.items()}

   # =======================================================================
   # enc_input = "Do you want to swim today?"
   enc_input = "I think there are a host of cats in my room but not dogs"
   pred = predict(model, enc_input, evoc, dvoc, dvoc_inv)
   print("".join(pred))
   # =======================================================================

   # # 用户输入
   # enc_input = "You've gained weight, haven't you?"
   # enc_input = " ".join(re.findall(r"[\w']+", enc_input))
   # enc_idx = torch.tensor([[evoc.get(tk, 1) for tk in enc_input.split()]], dtype=torch.long)# (batch_size=1, seq_len)
   
   # # print(enc_idx)



   # max_dec_tokens = 50

   # # 编码器
   # _, (h_n, c_n) = model.encoder(enc_idx)
   # h_n = torch.permute(h_n, (1, 0 ,2)).reshape(h_n.shape[1],-1).unsqueeze(0)
   # c_n = torch.permute(c_n, (1, 0 ,2)).reshape(c_n.shape[1],-1).unsqueeze(0)

   # # 解码器 输入shape:(1,1)
   # dec_input = torch.tensor([[dvoc["<BOS>"]]])
   # dec_tokens = []
   # while True:
   #    out = model.decoder_emb(dec_input)
   #    output, (h_n, c_n) = model.dec_rnn(out, (h_n, c_n))
   #    logits = model.dec_fc(output)
   #    next_token = logits.argmax(-1)

   #    if len(dec_tokens) > max_dec_tokens:
   #       break
   #    if dvoc_inv[next_token.item()] == "<EOS>":
   #       break
   #    dec_input = next_token
   #    dec_tokens.append(next_token.item())

   pass

