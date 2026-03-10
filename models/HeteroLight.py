import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
         Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        # [n_heads, batch_size, n_query, key_dim]
        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        # [n_heads, batch_size, targets_size, key_dim]
        K = torch.matmul(h_flat, self.w_key).view(shape_k)
        # [n_heads, batch_size, targets_size, value_dim]
        V = torch.matmul(h_flat, self.w_value).view(shape_v)

        # [n_heads, batch_size, n_query, targets_size]
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            # copy for n_heads times
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        # [n_heads, batch_size, n_query, targets_size]
        attention = torch.softmax(U, dim=-1)

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        # [n_heads, batch_size, n_query, value_dim]
        heads = torch.matmul(attention, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        # [batch_size, n_query, embedding_dim]
        return out


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """
        Sinusoid position encoding table
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttentionLayer(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        # self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Linear(512, embedding_dim))
        # self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        # h1 = h
        # h = self.normalization2(h)
        # h = self.feedForward(h)
        # h2 = h + h1
        return h


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)

        return tgt


class ActorNet(nn.Module):
    def __init__(self, input_dim, agent_dim, int_vec_dim, hidden_dim=128, vae_hidden_dim=20):
        super().__init__()
        self.input_dim = input_dim  # [max_num_movement, movement_feat]
        self.flat_dim = input_dim[0] * input_dim[1]
        self.flat_phase_vec_dim = input_dim[0]
        self.flat_int_vec_dim = int_vec_dim
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim
        self.vae_hidden_dim = vae_hidden_dim

        self.linear_s = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear_a = nn.Sequential(
            nn.Linear(self.flat_phase_vec_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        self.decoder = Decoder(embedding_dim=hidden_dim, n_head=4)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=False)
        self.gru_norm = Normalization(hidden_dim)

        # Add phase vec into the vae input
        self.linear_vae = nn.Sequential(
            nn.Linear(self.flat_dim + self.flat_int_vec_dim + self.flat_phase_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.linear_mean = nn.Linear(hidden_dim, vae_hidden_dim)
        self.linear_logvar = nn.Linear(hidden_dim, vae_hidden_dim)
        self.linear_recons = nn.Sequential(
            nn.Linear(vae_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim))

        self.policy_layer = nn.Sequential(
            nn.Linear(hidden_dim+vae_hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 1),
        )

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta):
        # Embedding
        # [b, n, feat_dim] -> [b, n, d]
        state = state.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        state_embedding = self.linear_s(state)
        batch_dim = state_embedding.size(0)
        # GRU
        state_res = state_embedding
        state_embedding, h_n = self.gru(state_embedding, h_n)
        state_embedding = state_embedding + state_res
        # [b*n, a_dim, d]
        phase_vector = phase_vector.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_phase_vec_dim)
        phase_vec_embedding = self.linear_a(phase_vector)
        # Decoder
        # [b*n, a_dim, d]
        state_embedding = state_embedding.reshape(batch_dim * self.agent_dim * num_meta, -1, self.hidden_dim)
        state_embedding = self.decoder(phase_vec_embedding, state_embedding)

        # VAE
        # [b*n, a_dim, d_s+d_i]
        int_vector = int_vector.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_int_vec_dim)
        state_vector = state.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_dim)
        state_vector = state_vector.expand(state_vector.size(0), int_vector.size(1), state_vector.size(2))
        # vae_input = torch.cat((state_vector, int_vector), -1)
        # Add phase vector into the vae input
        # [b*n, a_dim, d_s+d_i+d_p]
        vae_input = torch.cat((state_vector, int_vector, phase_vector), -1)
        vae_input = self.linear_vae(vae_input)
        mu = self.linear_mean(vae_input)
        logvar = self.linear_logvar(vae_input)
        # re-parameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # reconstruct s_(t+1）
        prediction = self.linear_recons(z)
        # [b, n, a_dim, input_dim]
        prediction = prediction.reshape(batch_dim, self.agent_dim*num_meta, -1, self.flat_dim)

        # [b, n, max_a_dim]
        state_embedding = torch.cat((state_embedding, z), -1)
        state_embedding = self.policy_layer(state_embedding).reshape(batch_dim, self.agent_dim * num_meta, -1)
        # Mask
        mask = mask.reshape(batch_dim, self.agent_dim*num_meta, -1)
        state_embedding[mask.bool()] = -np.inf
        policy = F.softmax(state_embedding, -1)

        return policy, h_n, prediction, mu.reshape(batch_dim, self.agent_dim*num_meta, -1, self.vae_hidden_dim), logvar.reshape(batch_dim, self.agent_dim*num_meta, -1, self.vae_hidden_dim)


class CriticNet(nn.Module):
    def __init__(self, input_dim, agent_dim, int_vec_dim, hidden_dim=128, vae_hidden_dim=20):
        super().__init__()
        self.input_dim = input_dim  # [max_num_movement, movement_feat]
        self.flat_dim = input_dim[0] * input_dim[1]
        self.flat_phase_vec_dim = input_dim[0]
        self.flat_int_vec_dim = int_vec_dim
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim
        self.vae_hidden_dim = vae_hidden_dim

        self.linear_s = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_a = nn.Sequential(
            nn.Linear(self.flat_phase_vec_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        self.decoder = Decoder(embedding_dim=hidden_dim, n_head=4)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=False)
        self.gru_norm = Normalization(hidden_dim)

        # Add phase vec into the vae input
        self.linear_vae = nn.Sequential(
            nn.Linear(self.flat_dim + self.flat_int_vec_dim + self.flat_phase_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_mean = nn.Linear(hidden_dim, vae_hidden_dim)
        self.linear_logvar = nn.Linear(hidden_dim, vae_hidden_dim)
        self.linear_recons = nn.Sequential(
            nn.Linear(vae_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim))

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim+vae_hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, 1),
        )

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta):
        # [b, n, d]
        state = state.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        state_embedding = self.linear_s(state)
        batch_dim = state_embedding.size(0)
        # GRU
        state_res = state_embedding
        state_embedding, h_n = self.gru(state_embedding, h_n)
        state_embedding = state_embedding + state_res
        # [b*n, a_dim, d]
        phase_vector = phase_vector.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_phase_vec_dim)
        phase_vec_embedding = self.linear_a(phase_vector)
        # [b*n, a_dim, d]
        state_embedding = state_embedding.reshape(batch_dim * self.agent_dim * num_meta, -1, self.hidden_dim)
        state_embedding = self.decoder(phase_vec_embedding, state_embedding)

        # VAE
        # [b*n, a_dim, d_s+d_i]
        int_vector = int_vector.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_int_vec_dim)
        state_vector = state.reshape(batch_dim * self.agent_dim * num_meta, -1, self.flat_dim)
        state_vector = state_vector.expand(state_vector.size(0), int_vector.size(1), state_vector.size(2))
        # vae_input = torch.cat((state_vector, int_vector), -1)
        # Add phase vector into the vae input
        # [b*n, a_dim, d_s+d_i+d_p]
        vae_input = torch.cat((state_vector, int_vector, phase_vector), -1)
        vae_input = self.linear_vae(vae_input)
        mu = self.linear_mean(vae_input)
        logvar = self.linear_logvar(vae_input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        prediction = self.linear_recons(z)
        # [b, n, a_dim, input_dim]
        prediction = prediction.reshape(batch_dim, self.agent_dim*num_meta, -1, self.flat_dim)

        # [b, n, 1]
        state_embedding = torch.cat((state_embedding, z), -1)
        value = self.value_layer(state_embedding).reshape(batch_dim, self.agent_dim * num_meta, -1)
        mask = mask.reshape(batch_dim, self.agent_dim*num_meta, -1)
        value[mask.bool()] = 0
        value = torch.sum(value, -1, keepdim=True)

        return value, h_n, prediction, mu.reshape(batch_dim, self.agent_dim*num_meta, -1, self.vae_hidden_dim), logvar.reshape(batch_dim, self.agent_dim*num_meta, -1, self.vae_hidden_dim)


class HeteroLight(nn.Module):
    def __init__(self, input_dim, agent_dim, int_vec_dim, actor_lr, critic_lr):
        super().__init__()
        self.input_dim = input_dim
        self.agent_dim = agent_dim

        self.actor_network = ActorNet(input_dim=self.input_dim, agent_dim=self.agent_dim, int_vec_dim=int_vec_dim)
        self.critic_network = CriticNet(input_dim=self.input_dim, agent_dim=self.agent_dim, int_vec_dim=int_vec_dim)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr)

    def forward(self, state, phase_vector, int_vector, mask, h_n, num_meta=1):
        return self.actor_network(state, phase_vector, int_vector, mask, h_n, num_meta)

    def forward_v(self, state, phase_vector, int_vector, mask, h_n, num_meta=1):
        return self.critic_network(state, phase_vector, int_vector, mask, h_n, num_meta)


if __name__ == '__main__':
    s = torch.randn((10, 5, 132))
    phase_vec = torch.randn((10, 5, 6, 22))
    int_vec = torch.randn((10, 5, 6, 56))
    phase_mask = torch.zeros((10, 5, 6))
    phase_mask[0, 0, 3:] = torch.ones(3)

    model = HeteroLight(input_dim=[22, 6], agent_dim=5, int_vec_dim=56, actor_lr=1e-5, critic_lr=1e-5)
    pi, _, pred1, mu1, logvar1 = model.forward(s, phase_vec, int_vec, phase_mask, None)
    v, _, pred2, mu2, logvar2 = model.forward_v(s, phase_vec, int_vec, phase_mask, None)
    print(pi.shape, v.shape, pred1.shape, pred2.shape, mu1.shape, mu2.shape)
