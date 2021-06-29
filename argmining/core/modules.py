from typing import Tuple

import torch
import torch.nn as nn
from allennlp.modules import Seq2VecEncoder


class BertCLSPooler(Seq2VecEncoder):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def get_input_dim(self) -> int:
        return self.hidden_dim

    def get_output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x, *args, **kwargs):
        # x - batch, seq_len, hidden
        return x[:, 0]  # batch, hidden


class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def generate_indexes(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_indexs = []
        end_indexs = []
        for i in range(1, length - 1):
            for j in range(i, length - 1):
                start_indexs.append(i)
                end_indexs.append(j)

        start_indexs = torch.LongTensor(start_indexs).to(self.device)
        end_indexs = torch.LongTensor(end_indexs).to(self.device)
        return start_indexs, end_indexs

    def forward(self, hidden_states, mask):
        # hidden_states - [batch_size, seq_len, d_model]
        # mask - [batch_size, seq_len]
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        length: int = mask.sum(dim=1).max().item()
        start_indexs, end_indexs = self.generate_indexes(length)

        W1_hi_emb = torch.index_select(
            W1_h, 1, start_indexs
        )  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = (
            W1_hi_emb
            + W2_hj_emb
            + (W3_hi_start_emb - W3_hi_end_emb)
            + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        )
        h_ij = torch.tanh(span)
        return h_ij


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num)
        # mask illegal span
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size)
        return H, a_ij
