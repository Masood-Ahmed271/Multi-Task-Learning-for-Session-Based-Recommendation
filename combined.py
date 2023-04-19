from torch import nn

class GRU4Rec(nn.Module):
    r"""
    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__()

        self.embedding_size = args.embedding_size
        self.vocab_size = args.num_items + 1
        self.n_layers = args.block_num
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        # define layers and loss
        self.item_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)

    '''
    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output
    '''

    def forward(self, item_seq):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, hidden = self.gru_layers(item_seq_emb_dropout)
        session_embedding = hidden[-1]  # Get the last hidden state as the session embedding
        gru_output = self.dense(gru_output)
        return session_embedding
    
    '''
Reference:
    [1]Jiaqi Ma et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD
    International Conference on Knowledge Discovery & Data Mining, pages 1930–1939, 2018.
Reference:
    https://github.com/busesese/MultiTaskModel
'''
import torch
import torch.nn as nn
from torch.nn import functional as F


class MMOE(nn.Module):
    """
    MMOE for CTCVR problem
    """

    def __init__(self, user_feature_dict, item_feature_dict, session_embedding, emb_dim=128, n_expert=2, mmoe_hidden_dim=128,
                 hidden_dim=[128, 128], dropouts=[0.5, 0.5], output_size=1, expert_activation=F.relu, num_task=2, device=None):
        """
        MMOE model input parameters
        :param user_feature_dict: user feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param item_feature_dict: item feature dict include: {feature_name: (feature_unique_num, feature_index)}
        :param emb_dim: int embedding dimension
        :param n_expert: int number of experts in mmoe
        :param mmoe_hidden_dim: mmoe layer input dimension
        :param hidden_dim: list task tower hidden dimension
        :param dropouts: list of task dnn drop out probability
        :param output_size: int task output size
        :param expert_activation: activation function like 'relu' or 'sigmoid'
        :param num_task: int default 2 multitask numbers
        """
        super(MMOE, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be not None")
        if isinstance(user_feature_dict, dict) is False or isinstance(item_feature_dict, dict) is False:
            raise Exception("input parameter user_feature_dict and item_feature_dict must be dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task
        self.session_embedding = session_embedding


        if device:
            self.device = device

        # embedding初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        hidden_size = emb_dim * (user_cate_feature_nums + item_cate_feature_nums) + \
                      (len(self.user_feature_dict) - user_cate_feature_nums) + (
                              len(self.item_feature_dict) - item_cate_feature_nums)

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size, mmoe_hidden_dim, n_expert), requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(mmoe_hidden_dim, n_expert), requires_grad=True)
        # gates
        self.gates = [torch.nn.Parameter(torch.rand(hidden_size, n_expert), requires_grad=True) for _ in
                      range(num_task)]
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = [torch.nn.Parameter(torch.rand(n_expert), requires_grad=True) for _ in range(num_task)]

        for i in range(self.num_task):
            setattr(self, 'task_{}_dnn'.format(i + 1), nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_hidden_{}'.format(j),
                                                                      nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_batchnorm_{}'.format(j),
                                                                      nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('ctr_dropout_{}'.format(j),
                                                                      nn.Dropout(dropouts[j]))
            getattr(self, 'task_{}_dnn'.format(i + 1)).add_module('task_last_layer',
                                                                  nn.Linear(hid_dim[-1], output_size))

    def forward(self, x, session_embedding):
        assert x.size()[1] == len(self.item_feature_dict) + len(self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)
        
        # hidden layer
        hidden = torch.cat([user_embed, item_embed], axis=1).float()  # batch * hidden_size
        hidden_with_session_embedding = torch.cat([hidden, session_embedding], axis=1)

        
        # mmoe
        experts_out = torch.einsum('ij, jkl -> ikl', hidden_with_session_embedding, self.experts)  # batch * mmoe_hidden_size * num_experts
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate = gate.to(self.device)
            gate_out = torch.einsum('ab, bc -> ac', hidden_with_session_embedding, gate)  # batch * num_experts
            if self.gates_bias:
                self.gates_bias[idx] = self.gates_bias[idx].to(self.device)
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)  # batch * 1 * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)  # batch * mmoe_hidden_size * num_experts
            outs.append(torch.sum(weighted_expert_output, 2))  # batch * mmoe_hidden_size

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, 'task_{}_dnn'.format(i + 1)):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs
