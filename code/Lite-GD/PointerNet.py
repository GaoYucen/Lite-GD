import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from config import get_config

params, _ = get_config()

if params.gpu and params.sys == 'win' and torch.cuda.is_available():
    USE_CUDA = True
    USE_MPS = False
    device = torch.device('cuda:0')
    print('Using GPU, %i devices.' % torch.cuda.device_count())
elif params.gpu and params.sys == 'mac':
    USE_CUDA = False
    USE_MPS = True
    device = torch.device('mps')
    print('Using MPS')
else:
    USE_CUDA = False
    USE_MPS = False
    device = torch.device('cpu')
    print('Using CPU')

class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2) #块和行交换

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        alpha = self.softmax(att)

        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)
        self.one_hot = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(params.seq_len):
            # print(_)
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # print(outs[0])
            # print(mask[0])
            # Masking selected inputs
            tmp_mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
            if (_ == 0):
                tmp_mask[:, 0] = 0
            elif (_ == 1) or (_ == 2):
                tmp_mask[:, 0] = 0
                tmp_mask[:, 11:-1] = 0
            elif (_ == 3) or (_ == 4):
                tmp_mask[:, 0:11] = 0
            cur_mask = torch.where((tmp_mask == 0) & (mask == 0), torch.zeros_like(tmp_mask), torch.max(tmp_mask, mask))
            masked_outs = outs * cur_mask
            # print(masked_outs[0])

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # 假设one_hot_pointers_can是你的张量，我们先创建一个全0的张量作为例子
            one_hot_pointers_can= self.one_hot.repeat(params.can_len).unsqueeze(0).repeat(batch_size, 1)
            # one_hot_pointers_can = torch.zeros((batch_size, params.can_len)).to(device)

            # 计算每一行中1的起始位置
            start_positions = ((indices - 1) / 5).int() * 5 + 1

            # 将每一行中1的起始位置及其后面的四个位置设为1
            for i in range(batch_size):

                if indices[i] == 0:
                    start = 0
                    end = 1
                else:
                    start = start_positions[i]
                    end = start + 5 if start + 5 <= 21 else 21  # 确保索引不超过列的数量
                one_hot_pointers_can[i, start:end] = 1
            # Update mask to ignore seen indices
            # print(one_hot_pointers_can[0])
            mask  = mask * (1 - one_hot_pointers_can)
            # print(mask[0])

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.embedding = nn.Linear(params.input_dimen, embedding_dim)
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        self.decoder = Decoder(embedding_dim, hidden_dim)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)
        if self.bidir:
            decoder_hidden0 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), dim=-1),
                               torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return  outputs, pointers