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
        return x[:, 0]


class SelfExplainingLayer(nn.Module):
    """
    Воспроизведение слоя интерпретации из статьи
    https://arxiv.org/pdf/2012.01786v2.pdf

    Возвращает взвешенную сумму всех спанов текста и
    веса всех спанов текста (альфы)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает: h_weighted - взвешенное представление всех спанов [batch_size, d_model] и
        веса спанов alpha - [batch_size, num_spans].
        :param h: - скрытое представление текста [batch_size, seq_len, d_model]
        :param mask: - маска текста [batch_size, seq_len]
        """
        pass


