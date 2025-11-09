import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from langconv import Converter
from nltk import word_tokenize
from torch.autograd import Variable
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import csv

torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的目录
os.makedirs('ablation_results', exist_ok=True)
os.makedirs('ablation_images', exist_ok=True)

# 配置matplotlib使用英文标签
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']


PAD = 0  # padding占位符的索引
UNK = 1  # 未登录词标识符的索引
BATCH_SIZE = 128  # 批次大小
EPOCHS = 20  # 训练轮数
LAYERS = 6  # transformer中encoder、decoder层数
H_NUM = 8  # 多头注意力个数
D_MODEL = 256  # 输入、输出词向量维数
D_FF = 1024  # feed forward全连接层维数
DROPOUT = 0.1  # dropout比例
MAX_LENGTH = 60  # 语句最大长度

TRAIN_FILE = 'dataset/en-cn/train.txt'  # 训练集
DEV_FILE = "dataset/en-cn/test.txt"  # 验证集
SAVE_FILE = 'save/model.pt'  # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 消融实验配置
ABLATION_EXPERIMENTS = {
    'baseline': {
        'description': '基线',
        'positional_encoding': 'sinusoidal',
        'num_heads': 8,
        'use_residual': True,
        'use_layernorm': True,
        'ffn_activation': 'relu'
    },
    'no_positional_encoding': {
        'description': '无位置编码',
        'positional_encoding': 'none',
        'num_heads': 8,
        'use_residual': True,
        'use_layernorm': True,
        'ffn_activation': 'relu'
    },
    'single_head_attention': {
        'description': '单头注意力',
        'positional_encoding': 'sinusoidal',
        'num_heads': 1,
        'use_residual': True,
        'use_layernorm': True,
        'ffn_activation': 'relu'
    },
    'no_residual_connection': {
        'description': '无残差连接',
        'positional_encoding': 'sinusoidal',
        'num_heads': 8,
        'use_residual': False,
        'use_layernorm': True,
        'ffn_activation': 'relu'
    },
    'no_layer_norm': {
        'description': '无层归一化',
        'positional_encoding': 'sinusoidal',
        'num_heads': 8,
        'use_residual': True,
        'use_layernorm': False,
        'ffn_activation': 'relu'
    },
    'learned_positional_encoding': {
        'description': '学习的位置编码',
        'positional_encoding': 'learned',
        'num_heads': 8,
        'use_residual': True,
        'use_layernorm': True,
        'ffn_activation': 'relu'
    }
}


def seq_padding(X, padding=PAD):
    #按批次（batch）对数据填充、长度对齐
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def cht_to_chs(sent):
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent


class PrepareData:
    def __init__(self, train_file, dev_file):
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)
        self.train_en, self.train_cn = self.word2id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)
        self.train_data = self.split_batch(self.train_en, self.train_cn, BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, BATCH_SIZE)

    def load_data(self, path):
        en = []
        cn = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split("\t")
                sent_en = sent_en.lower()
                sent_cn = cht_to_chs(sent_cn)
                sent_en = ["BOS"] + word_tokenize(sent_en) + ["EOS"]
                sent_cn = ["BOS"] + [char for char in sent_cn] + ["EOS"]
                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_dict(self, sentences, max_words=5e4):
        word_count = Counter([word for sent in sentences for word in sent])
        ls = word_count.most_common(int(max_words))
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        length = len(en)
        out_en_ids = [[en_dict.get(word, UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, UNK) for word in sent] for sent in cn]

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[idx] for idx in sorted_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sorted_index]
        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
        return batches


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0.0, max_len, device=DEVICE)
        position.unsqueeze_(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)
        pe[:, 0:: 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1:: 2] = torch.cos(torch.mm(position, div_term))
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += Variable(self.pe[:, : x.size(1), :], requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps)
        return self.a_2 * x + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        return x + x_


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, trg=None, pad=PAD):
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, : -1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


# 消融实验专用组件
class LearnedPositionalEncoding(nn.Module):
    # 学习的位置编码（替代正弦位置编码）

    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NoResidualNoNormEncoderLayer(nn.Module):
    # 无残差连接和无层归一化的编码器层

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(NoResidualNoNormEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        # 无归一化，无残差连接
        x = self.dropout(self.self_attn(x, x, x, mask))
        x = self.dropout(self.feed_forward(x))
        return x


class NoResidualEncoderLayer(nn.Module):
    # 无残差连接的编码器层

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(NoResidualEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        # 有归一化，无残差连接
        x = self.norm(x)
        x = self.dropout(self.self_attn(x, x, x, mask))
        x = self.norm(x)
        x = self.dropout(self.feed_forward(x))
        return x


class NoLayerNormEncoderLayer(nn.Module):
    # 无层归一化的编码器层

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(NoLayerNormEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        # 无归一化，有残差连接
        attn_output = self.dropout(self.self_attn(x, x, x, mask))
        x = x + attn_output  # 残差连接

        ffn_output = self.dropout(self.feed_forward(x))
        x = x + ffn_output  # 残差连接
        return x


def make_ablation_model(src_vocab, tgt_vocab, config, N=6, d_model=512, d_ff=2048, dropout=0.1):
    # 根据消融实验配置创建模型
    c = copy.deepcopy

    # 根据配置选择位置编码
    if config['positional_encoding'] == 'sinusoidal':
        position = PositionalEncoding(d_model, dropout).to(DEVICE)
    elif config['positional_encoding'] == 'learned':
        position = LearnedPositionalEncoding(d_model, dropout).to(DEVICE)
    else:  # 'none'
        position = nn.Identity().to(DEVICE)  # 无位置编码

    # 根据配置设置注意力头数
    h = config['num_heads']
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)

    # 根据配置选择编码器层类型
    if not config['use_residual'] and not config['use_layernorm']:
        encoder_layer = NoResidualNoNormEncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE)
    elif not config['use_residual']:
        encoder_layer = NoResidualEncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE)
    elif not config['use_layernorm']:
        encoder_layer = NoLayerNormEncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE)
    else:
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE)

    model = Transformer(
        Encoder(encoder_layer, N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), position),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), position),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)



class AblationExperiment:
    def __init__(self, config, experiment_name):
        self.config = config
        self.name = experiment_name
        self.results = {}
        self.model = None
        self.model_save_path = f'save/model_ablation_{experiment_name}.pt'

    def setup_model(self, src_vocab, tgt_vocab):
        print(f"Setting up ablation experiment: {self.name}")
        print(f"Description: {self.config['description']}")

        # 使用全局配置参数
        self.model = make_ablation_model(src_vocab, tgt_vocab, self.config,
                                         N=LAYERS, d_model=D_MODEL, d_ff=D_FF, dropout=DROPOUT)
        return self.model

    def train_model(self, data, epochs=10):
        # 训练模型
        print(f"Training ablation model: {self.name}")

        ablation_train_losses = []
        ablation_val_losses = []

        criterion = LabelSmoothing(len(data.cn_word_dict), padding_idx=0, smoothing=0.0)
        optimizer = NoamOpt(D_MODEL, 1, 2000,
                            torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        best_dev_loss = 1e5

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = run_epoch(data.train_data, self.model,
                                   SimpleLossCompute(self.model.generator, criterion, optimizer),
                                   epoch, is_training=True)
            ablation_train_losses.append(train_loss)

            # 验证
            self.model.eval()
            val_loss = run_epoch(data.dev_data, self.model,
                                 SimpleLossCompute(self.model.generator, criterion, None),
                                 epoch, is_training=False)
            ablation_val_losses.append(val_loss)

            if val_loss < best_dev_loss:
                torch.save(self.model.state_dict(), self.model_save_path)
                best_dev_loss = val_loss

            print(f'Ablation {self.name} - Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        self.results['train_losses'] = ablation_train_losses
        self.results['val_losses'] = ablation_val_losses
        self.results['best_val_loss'] = best_dev_loss

        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.model_save_path))

        return self.results

    def evaluate_model(self, data, max_samples=100):
        # 评估模型
        print(f"Evaluating ablation model: {self.name}")

        results = evaluate_with_metrics_ablation(data, self.model, max_samples, self.name)
        self.results.update(results)

        # 计算关键指标
        avg_bleu = np.mean(results['bleu_scores'])
        exact_match_rate = results['exact_matches'] / results['total_samples']

        self.results['avg_bleu'] = avg_bleu
        self.results['exact_match_rate'] = exact_match_rate

        print(f"Ablation {self.name} - Avg BLEU: {avg_bleu:.4f}, Exact Match: {exact_match_rate:.4f}")

        return self.results


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # 确保返回 Python 数值，而不是张量
        return loss.item()  # 直接使用 loss.item() 获取标量值


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def run_epoch(data, model, loss_compute, epoch, is_training=True):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens.item()
        tokens += batch.ntokens.item()

        if i % 50 == 1:
            elapsed = time.time() - start
            mode = "Training" if is_training else "Validation"
            print(
                f"Epoch {epoch} {mode} Batch: {i - 1} Loss: {loss / batch.ntokens.item():.4f} Tokens per Sec: {tokens / elapsed / 1000.:.2f}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def plot_positional_encoding():
    emb_dim = 64
    max_seq_len = 100
    seq_len = 20

    pe = PositionalEncoding(emb_dim, 0, max_seq_len)
    positional_encoding = pe(torch.zeros(1, seq_len, emb_dim, device=DEVICE))

    # 热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(positional_encoding.squeeze().cpu().numpy())
    plt.xlabel("Position")
    plt.ylabel("Dimension")
    plt.title("Positional Encoding Heatmap")
    plt.savefig('images/positional_encoding_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 线图
    plt.figure(figsize=(10, 6))
    y = positional_encoding.cpu().numpy()
    plt.plot(np.arange(seq_len), y[0, :, 0:64:8], ".")
    plt.legend(["dim %d" % p for p in [0, 7, 15, 31, 63]])
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Positional Encoding Values")
    plt.savefig('images/positional_encoding_values.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_mask():
    plt.figure(figsize=(6, 6))
    plt.imshow(subsequent_mask(20)[0].numpy())
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Subsequent Mask")
    plt.colorbar()
    plt.savefig('images/attention_mask.png', dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_with_metrics_ablation(data, model, max_samples=100, experiment_name=""):
    results = {
        'samples': [],
        'bleu_scores': [],
        'exact_matches': 0,
        'total_samples': 0,
        'experiment_name': experiment_name
    }

    with torch.no_grad():
        sample_count = min(max_samples, len(data.dev_en))
        print(f"Evaluating ablation experiment '{experiment_name}' on {sample_count} samples...")
        # 检查是否意外使用了训练数据
        if hasattr(data, 'train_en') and len(data.dev_en) == len(data.train_en):
            print("警告: 测试集与训练集大小相同，可能存在数据混淆!")

        for i in range(sample_count):
            # 获取源语句
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])

            cn_reference = "".join([data.cn_index_dict[w] for w in data.dev_cn[i] if
                                    w not in [data.cn_word_dict['BOS'], data.cn_word_dict['EOS']]])
            reference_chars = list(cn_reference)  # 字符级分词

            # 模型推理
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            src = src.unsqueeze(0)
            src_mask = (src != 0).unsqueeze(-2)

            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH,
                                start_symbol=data.cn_word_dict["BOS"])

            # 获取翻译结果
            translation = []
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
            translation_str = "".join(translation)
            hypothesis_chars = list(translation_str)  # 字符级分词

            # 计算BLEU分数
            bleu_score = calculate_bleu_ablation(reference_chars, hypothesis_chars)

            # 检查是否完全匹配
            exact_match = 1 if translation_str == cn_reference else 0

            results['samples'].append({
                'source': en_sent,
                'reference': cn_reference,
                'translation': translation_str,
                'bleu_score': bleu_score,
                'exact_match': exact_match
            })
            results['bleu_scores'].append(bleu_score)
            results['exact_matches'] += exact_match
            results['total_samples'] += 1

            if i < 5:  # 只打印前5个样本的详细信息
                print(f"\nSample {i + 1}:")
                print(f"Source: {en_sent}")
                print(f"Reference: {cn_reference}")
                print(f"Translation: {translation_str}")
                print(f"BLEU Score: {bleu_score:.4f}")
                print(f"Exact Match: {'Yes' if exact_match else 'No'}")
                print("-" * 50)

    return results


def calculate_bleu_ablation(reference, hypothesis):
    # 计算BLEU分数（消融实验专用）
    smooth = SmoothingFunction()
    # 使用更适合消融实验的权重和平滑方法
    return sentence_bleu([reference], hypothesis,
                         weights=(0.4, 0.3, 0.2, 0.1),  # 侧重1-gram和2-gram
                         smoothing_function=smooth.method4)


def plot_evaluation_results(results):
    # 绘制评估结果图（消融实验专用）
    # 确保目录存在
    os.makedirs('ablation_images', exist_ok=True)

    # BLEU分数分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(results['bleu_scores'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.title('BLEU Score Distribution')
    plt.grid(True, alpha=0.3)

    # 准确率饼图
    plt.subplot(1, 2, 2)
    exact_match_rate = results['exact_matches'] / results['total_samples']
    labels = ['Exact Match', 'Not Exact Match']
    sizes = [exact_match_rate, 1 - exact_match_rate]
    colors = ['lightgreen', 'lightcoral']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Exact Match Rate')

    plt.tight_layout()
    plt.savefig('ablation_images/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存详细结果到JSON文件
    with open('ablation_results/evaluation_details.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_ablation_study(data, src_vocab, tgt_vocab, ablation_configs, epochs=10):
    # 运行消融实验研究
    print("\n" + "=" * 60)
    print("STARTING ABLATION STUDY")
    print("=" * 60)

    ablation_results = {}

    # 确保消融实验目录存在
    os.makedirs('ablation_results', exist_ok=True)
    os.makedirs('ablation_images', exist_ok=True)

    # 运行每个消融实验
    for exp_name, config in ablation_configs.items():
        print(f"\n{'=' * 50}")
        print(f"Running ablation experiment: {exp_name}")
        print(f"Description: {config['description']}")
        print(f"{'=' * 50}")

        # 创建实验实例
        experiment = AblationExperiment(config, exp_name)

        # 设置模型
        model = experiment.setup_model(src_vocab, tgt_vocab)

        # 训练模型（使用较少的epochs）
        train_results = experiment.train_model(data, epochs=epochs)

        # 评估模型
        eval_results = experiment.evaluate_model(data, max_samples=100)

        # 保存结果
        ablation_results[exp_name] = experiment.results

        print(f"Completed ablation experiment: {exp_name}")

    return ablation_results


def plot_ablation_results(ablation_results, ablation_configs):
    # 创建多个子图对比不同实验的结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. BLEU分数对比
    exp_names = list(ablation_results.keys())
    exp_descriptions = [ablation_configs[name]['description'] for name in exp_names]
    bleu_scores = [ablation_results[name]['avg_bleu'] for name in exp_names]

    axes[0, 0].bar(range(len(exp_names)), bleu_scores, color='skyblue')
    axes[0, 0].set_title('BLEU Score Comparison')
    axes[0, 0].set_ylabel('BLEU Score')
    axes[0, 0].set_xticks(range(len(exp_names)))
    axes[0, 0].set_xticklabels(exp_descriptions, rotation=45, ha='right')

    # 2. 精确匹配率对比
    exact_match_rates = [ablation_results[name]['exact_match_rate'] for name in exp_names]
    axes[0, 1].bar(range(len(exp_names)), exact_match_rates, color='lightgreen')
    axes[0, 1].set_title('Exact Match Rate Comparison')
    axes[0, 1].set_ylabel('Exact Match Rate')
    axes[0, 1].set_xticks(range(len(exp_names)))
    axes[0, 1].set_xticklabels(exp_descriptions, rotation=45, ha='right')

    # 3. 验证损失曲线对比
    for exp_name in exp_names:
        val_losses = ablation_results[exp_name]['val_losses']
        axes[1, 0].plot(val_losses, label=ablation_configs[exp_name]['description'], linewidth=2)
    axes[1, 0].set_title('Validation Loss Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 训练损失曲线对比
    for exp_name in exp_names:
        train_losses = ablation_results[exp_name]['train_losses']
        axes[1, 1].plot(train_losses, label=ablation_configs[exp_name]['description'], linewidth=2)
    axes[1, 1].set_title('Training Loss Curves')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ablation_images/ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存详细结果
    with open('ablation_results/ablation_study.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # 创建结果汇总表
    summary_data = []
    for exp_name in exp_names:
        summary_data.append({
            'Experiment': exp_name,
            'Description': ablation_configs[exp_name]['description'],
            'BLEU Score': f"{ablation_results[exp_name]['avg_bleu']:.4f}",
            'Exact Match Rate': f"{ablation_results[exp_name]['exact_match_rate']:.4f}",
            'Best Val Loss': f"{ablation_results[exp_name]['best_val_loss']:.4f}"
        })

    # 打印结果表
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    for item in summary_data:
        print(f"{item['Experiment']:25} | {item['Description']:30} | "
              f"BLEU: {item['BLEU Score']:6} | Exact Match: {item['Exact Match Rate']:6} | "
              f"Val Loss: {item['Best Val Loss']:6}")

    # 保存汇总表
    with open('ablation_results/ablation_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)


def train_model(self, data, epochs=10):
    # 训练模型（使用较少的epochs进行消融实验）
    print(f"Training ablation model: {self.name}")

    ablation_train_losses = []
    ablation_val_losses = []

    criterion = LabelSmoothing(len(data.cn_word_dict), padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(D_MODEL, 1, 2000,
                        torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    best_dev_loss = 1e5

    for epoch in range(epochs):
        # 训练
        self.model.train()
        train_loss = run_epoch(data.train_data, self.model,
                               SimpleLossCompute(self.model.generator, criterion, optimizer),
                               epoch, is_training=True)
        ablation_train_losses.append(train_loss)

        # 验证
        self.model.eval()
        val_loss = run_epoch(data.dev_data, self.model,
                             SimpleLossCompute(self.model.generator, criterion, None),
                             epoch, is_training=False)
        ablation_val_losses.append(val_loss)

        if val_loss < best_dev_loss:
            torch.save(self.model.state_dict(), self.model_save_path)
            best_dev_loss = val_loss

        print(f'Ablation {self.name} - Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

    self.results['train_losses'] = ablation_train_losses
    self.results['val_losses'] = ablation_val_losses
    self.results['best_val_loss'] = best_dev_loss

    # 加载最佳模型
    self.model.load_state_dict(torch.load(self.model_save_path))

    return self.results


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 确保使用模型的解码方法
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # 创建正确的掩码
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)

        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(tgt_mask))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()  # 使用 item() 获取标量值

        # 使用正确的 EOS 检查
        if next_word == data.cn_word_dict["EOS"]:
            break

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    return ys


if __name__ == "__main__":
    # 数据预处理
    data = PrepareData(TRAIN_FILE, DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print(f"Source vocabulary: {src_vocab}")
    print(f"Target vocabulary: {tgt_vocab}")

    print("\n" + "=" * 60)
    print("STARTING ABLATION STUDY (ONLY)")
    print("=" * 60)

    ablation_start = time.time()

    # 运行消融实验
    ablation_results = run_ablation_study(data, src_vocab, tgt_vocab, ABLATION_EXPERIMENTS, epochs=20)

    # 绘制和保存消融实验结果
    plot_ablation_results(ablation_results, ABLATION_EXPERIMENTS)

    ablation_time = time.time() - ablation_start
    print(f"\nAblation study completed in {ablation_time:.2f} seconds!")
    print("Ablation results saved to 'ablation_results/' and 'ablation_images/' directories")


    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Summary of saved results:")
    print("- Ablation study results: ablation_results/ and ablation_images/")
    print("=" * 60)

