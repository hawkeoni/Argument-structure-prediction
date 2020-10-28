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
