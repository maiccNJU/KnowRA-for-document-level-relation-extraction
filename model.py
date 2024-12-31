from typing import List, Tuple

import hydra.utils
import math
import torch
import torch.nn as nn
from opt_einsum import contract
from torch import Tensor

from long_seq import process_long_input, process_long_input_longformer
from losses import AFLoss, NCRLoss
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import RelGraphConv
from transformers import AutoModel, AutoConfig
import numpy as np


class AxialTransformer_by_entity(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attns = nn.ModuleList(
            [AxialAttention(dim=self.emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions, ) for i
             in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])

    def forward(self, x):
        for idx in range(self.num_layers):
            x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
            x = self.ffns[idx](x)
            x = self.ffn_dropouts[idx](x)
            x = self.lns[idx](x)
        return x


class NoAxialTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class AxialEntityTransformer(nn.Module):
    def __init__(self, emb_size=768, dropout=0.1, num_layers=2, dim_index=-1, heads=8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_img_transformer = AxialImageTransformer()
        self.axial_attns = nn.ModuleList(
            [AxialAttention(dim=self.emb_size, dim_index=dim_index, heads=heads, num_dimensions=num_dimensions, ) for i
             in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])

    def forward(self, x):
        for idx in range(self.num_layers):
            x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
            x = self.ffns[idx](x)
            x = self.ffn_dropouts[idx](x)
            x = self.lns[idx](x)
        return x


class GCNGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 *,
                 weight=True,
                 bias=True,
                 activation=nn.Tanh(),
                 self_loop=True,
                 dropout=0.):
        super(GCNGraphConvLayer, self).__init__()
        num_bases = len(rel_names)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class GATGraphConvLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names, fp, ap, residual, activation):
        super(GATGraphConvLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feat, out_feat, num_heads=1, feat_drop=fp, attn_drop=ap, residual=residual, activation=activation)
            for rel in rel_names
        })

    def forward(self, g, inputs):
        hs = self.conv(g, inputs)
        return {ntype: h.squeeze(1) for ntype, h in hs.items()}


class GATGraphConv(nn.Module):
    def __init__(self, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation, num_layers):
        super().__init__()
        self.graph_conv = nn.ModuleList([
            GATGraphConvLayer(hidden_dim, hidden_dim, edge_types, feat_drop, attn_drop, residual, activation)
            for _ in range(num_layers)
        ])

    def forward(self, graph, feat):
        for graph_layer in self.graph_conv:
            feat = graph_layer(graph, {'node': feat})['node']
        return feat


class NoGraphConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        return torch.zeros_like(feat)


class KGConv(nn.Module):
    def __init__(self, num_rels, feat_dim, emb_dim):
        super().__init__()
        self.num_rels = num_rels
        self.rel_projection = nn.Embedding(num_rels, emb_dim)
        self.triple_projection = nn.Linear(2 * feat_dim + emb_dim, feat_dim)

    def forward(self, graph, inputs):
        graph.nodes['node'].data['ent'] = inputs
        rel_embeddings = self.rel_projection(torch.arange(self.num_rels, device=inputs.device))
        for rel_id in range(self.num_rels):
            graph.edges['node', rel_id, 'node'].data['rel'] = \
                rel_embeddings[rel_id].unsqueeze(0).expand(graph.num_edges(rel_id), -1)

        def message_func(edges):
            sub, rel, obj = edges.src['ent'], edges.data['rel'], edges.dst['ent']
            triple = torch.cat([sub, obj, rel], dim=-1)
            return {"trp_emb": self.triple_projection(triple)}

        def reduce_func(nodes):
            return {"ent_": torch.tanh(nodes.mailbox["trp_emb"].sum(dim=1))}

        graph.multi_update_all({('node', rel_id, 'node'): (message_func, reduce_func)
                                for rel_id in range(self.num_rels)})
        return graph.nodes['node'].data.pop('ent_')


class KgRelGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_layers, regularizer, num_bases, dropout):
        super().__init__()
        self.conv = nn.ModuleList([
            RelGraphConv(in_feat, out_feat, num_rels, regularizer=regularizer, num_bases=num_bases,
                         dropout=dropout, activation=nn.Tanh()) for _ in range(num_layers)
        ])

    def forward(self, graph, feat, etype):
        for layer in self.conv:
            feat = layer(graph, feat, etype)
        return feat


def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = torch.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res


def batched_l1_dist(a, b):
    res = torch.cdist(a, b, p=1)
    return res


class TransEScore(nn.Module):
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else:  # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': torch.sigmoid(self.gamma - torch.norm(score, p=self.dist_ord, dim=-1)),
                'trans': head + rel}

    def reduce_func(self, nodes):
        s, t = nodes.mailbox['score'], nodes.mailbox['trans']
        h = (s.unsqueeze(-1) * t).sum(1)
        return {'h': h}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)

        return self.gamma - torch.norm(score, p=self.dist_ord, dim=-1)

    def forward(self, g):
        # g.apply_edges(lambda edges: self.edge_func(edges))
        g.update_all(lambda edges: self.edge_func(edges), lambda nodes: self.reduce_func(nodes))
        return g.ndata.pop('h')


class DistMultScore(nn.Module):
    def __init__(self):
        super(DistMultScore, self).__init__()

    def score_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        return {'dr': score.sum(-1)}

    def edge_func(self, edges):
        head = edges.src['emb']
        rel = edges.data['emb']
        score_logits = edges.data['dr']
        trans = head * rel
        return {'score': torch.sigmoid(score_logits), 'trans': trans}

    def reduce_func(self, nodes):
        s, t = nodes.mailbox['score'], nodes.mailbox['trans']
        h = (s.unsqueeze(-1) * t).sum(1)
        return {'h': h}

    def forward(self, g):
        # g.apply_edges(lambda edges: self.edge_func(edges))
        g.apply_edges(lambda edges: self.score_func(edges))
        g.update_all(lambda edges: self.edge_func(edges), lambda nodes: self.reduce_func(nodes))
        return g.ndata.pop('h'), g.edata.pop('dr')


class TypeEmbedding(nn.Module):
    def __init__(self, num_rels, num_bases, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coeff = nn.Parameter(torch.Tensor(num_rels, num_bases))
        self.W = nn.Parameter(torch.Tensor(num_bases, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('tanh'))
            nn.init.uniform_(self.W, -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

    def get_weight(self):
        return self.coeff @ self.W

    def forward(self, etype):
        w = self.get_weight()
        return w[etype.long()]


class KGEmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim, num_rels, num_bases, dropout, score_func, activation):
        super().__init__()
        self.type_emb = TypeEmbedding(num_rels, num_bases, hidden_dim)
        self.score_func = score_func
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat, etypes):
        if graph.num_edges() == 0:
            return torch.zeros_like(feat), torch.zeros_like(etypes), False
        with graph.local_scope():
            graph.ndata['emb'] = feat
            graph.edata['etype'] = etypes
            graph.apply_edges(lambda edges: {'emb': self.type_emb(edges.data['etype'])})
            h, score = self.score_func(graph)
            return self.dropout(self.activation(h)), score, True


class NoKGEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat, etypes):
        return torch.zeros_like(feat), torch.zeros_like(etypes), False


class KGEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_rels, num_layers, num_bases, dropout, score_func):
        super().__init__()
        self.conv = nn.ModuleList([
            KGEmbeddingLayer(hidden_dim, num_rels, num_bases, dropout, score_func, nn.Tanh())
            for _ in range(num_layers)
        ])

    def forward(self, graph, feat, etype):
        for layer in self.conv:
            feat, score = layer(graph, feat, etype)
        return feat, score


class DocREModel(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 max_seq_length,
                 transformer_type,
                 tokenizer,
                 graph_conv,
                 residual,
                 coref,
                 num_class,
                 block_size,
                 kg_conv,
                 axial_conv,
                 kg_loss_weight,
                 loss_fnt):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length
        self.config.cls_token_id = tokenizer.cls_token_id
        self.config.sep_token_id = tokenizer.sep_token_id
        self.config.transformer_type = transformer_type
        self.config.model_max_len = self.config.max_position_embeddings
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.config.hidden_size
        self.emb_size = self.hidden_size
        self.block_size = block_size
        self.rel_loss_fnt = loss_fnt
        self.kg_loss_weight = kg_loss_weight
        self.kg_loss_fnt = nn.BCEWithLogitsLoss()
        self.head_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.tail_extractor = nn.Linear(2 * self.hidden_size, self.emb_size)
        self.projection = nn.Linear(self.emb_size * block_size, self.hidden_size, bias=False)
        self.classifier = nn.Linear(self.hidden_size, num_class)
        if isinstance(axial_conv, NoAxialTransformer):  #   -w/o axial attention
            self.axial_conv = axial_conv
        else:
            self.axial_conv = axial_conv(emb_size=self.hidden_size)
        if isinstance(graph_conv, NoGraphConv):   #  -w/o graph neural network络
            self.graph_conv = graph_conv
        else:
            self.graph_conv = graph_conv(hidden_dim=self.hidden_size)
        if isinstance(kg_conv, NoKGEmbeddingLayer):  # -w/o knowledge augmentation method
            self.kg_conv = kg_conv
        else:
            self.kg_conv = kg_conv(hidden_dim=self.hidden_size)
        self.residual = residual
        assert coref in {'gated', 'e_context'}
        self.coref = coref
        # self.emb_size = emb_size
        self.block_size = block_size
        self.num_class = num_class
        # self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert" or config.transformer_type == 'deberta':
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        elif config.transformer_type == 'longformer':
            return process_long_input_longformer(self.model, input_ids, attention_mask)
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens, 512)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, hts, sent_pos, entity_pos, coref_pos, mention_pos, men_graphs, ent_graphs, etypes, ne):
        offset = 1 if self.config.transformer_type in ["bert", "roberta", "longformer", "deberta"] else 0
        batch_size, num_neads, seq_len, seq_len = attention.size()
        batch_size, seq_len, hidden_size = sequence_output.size()
        hss, rss, tss = [], [], []
        device = sequence_output.device
        n_e = ne

        feats = []
        nms, nss, nes = [len(m) for m in mention_pos], [len(s) for s in sent_pos], [len(e) for e in entity_pos]
        for i in range(batch_size):
            doc_emb = sequence_output[i][0].unsqueeze(0)

            mention_embs = sequence_output[i, mention_pos[i] + offset]

            sentence_embs = [torch.logsumexp(sequence_output[i, offset + sent_pos[0]:offset + sent_pos[1]], dim=0) for
                             sent_pos in sent_pos[i]]
            sentence_embs = torch.stack(sentence_embs)

            all_embs = torch.cat([doc_emb, mention_embs, sentence_embs], dim=0)
            feats.append(all_embs)
        feats = torch.cat(feats, dim=0)
        assert len(feats) == batch_size + sum(nms) + sum(nss)
        feats = self.graph_conv(men_graphs, feats)

        cur_idx = 0
        batch_entity_embs, batch_entity_atts = [], []
        for i in range(batch_size):  # 
            entity_embs, entity_atts = [], []

            men_idx = -1
            for e_id, e in enumerate(entity_pos[i]):  # 
                if len(e) > 1:  # 
                    e_emb, g_emb, e_att = [], [], []
                    for start, end in e:  # 
                        men_idx += 1
                        if start + offset < seq_len:
                            e_emb.append(sequence_output[i, start + offset])
                            g_emb.append(feats[cur_idx + 1 + men_idx])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        if self.residual:  # 
                            e_emb = torch.stack(e_emb) + torch.stack(g_emb)
                        else:
                            e_emb = torch.stack(g_emb)
                        if self.coref == 'gated':
                            att = torch.stack(e_att).mean(0).sum(0)
                            gate_score = att / att.sum()
                            coref_emb = []
                            for start, end in coref_pos[i][e_id]:
                                coref_emb.append((gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                            if coref_emb:  # 
                                e_emb = torch.cat([e_emb, torch.stack(coref_emb)])
                        e_emb = torch.logsumexp(e_emb, dim=0)
                        if self.coref == 'e_context':
                            for start, end in coref_pos[i][e_id]:
                                e_att.append(attention[i, :, start:end].mean(1))
                        e_att = torch.stack(e_att).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_neads, seq_len).to(attention)
                else:
                    start, end = e[0]
                    men_idx += 1
                    if start + offset < seq_len:
                        if self.residual:
                            e_emb = sequence_output[i, start + offset] + feats[cur_idx + 1 + men_idx]
                        else:
                            e_emb = feats[cur_idx + 1 + men_idx]
                        if self.coref == 'gated':
                            e_att = attention[i, :, start + offset]
                            att = e_att.sum(0)
                            gate_score = att / att.sum()
                            coref_emb = []
                            for start, end in coref_pos[i][e_id]:
                                coref_emb.append((gate_score[start:end].unsqueeze(-1) * sequence_output[i, start:end]).sum(0))
                            if coref_emb:
                                e_emb = torch.cat([e_emb.unsqueeze(0), torch.stack(coref_emb)])
                                e_emb = torch.logsumexp(e_emb, dim=0)
                        else:  # coref == 'e_context'
                            if not coref_pos[i][e_id]:
                                e_att = attention[i, :, start + offset]
                            else:
                                e_att = [attention[i, :, start + offset]]
                                for start, end in coref_pos[i][e_id]:
                                    e_att.append(attention[i, :, start:end].mean(1))
                                e_att = torch.stack(e_att).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_neads, seq_len).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            cur_idx += 1 + nms[i] + nss[i]
            entity_embs = torch.stack(entity_embs)
            entity_atts = torch.stack(entity_atts)
            batch_entity_embs.append(entity_embs)
            batch_entity_atts.append(entity_atts)

        all_entity_embs = torch.cat(batch_entity_embs)
        kg_feats, kg_score, kg_flag = self.kg_conv(ent_graphs, all_entity_embs, etypes)
        cur_idx = 0
        for i in range(batch_size):
            entity_embs = batch_entity_embs[i] + kg_feats[cur_idx:cur_idx + nes[i]]
            cur_idx += nes[i]
            entity_atts = batch_entity_atts[i]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            pad_hs = torch.zeros((n_e, n_e, hidden_size)).to(device)
            pad_ts = torch.zeros((n_e, n_e, hidden_size)).to(device)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, hidden_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, hidden_size)

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            # ht_att = (h_att * t_att).mean(1)
            m = torch.nn.Threshold(0, 0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)

            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros(n_e, n_e, hidden_size).to(device)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, hidden_size)

            hss.append(pad_hs)
            rss.append(pad_rs)
            tss.append(pad_ts)

        hss = torch.stack(hss)
        tss = torch.stack(tss)
        rss = torch.stack(rss)
        return hss, rss, tss, kg_score, kg_flag

    def forward(self,
                input_ids,
                attention_mask,
                hts,
                sent_pos,
                entity_pos,
                coref_pos,
                mention_pos,
                entity_types,
                men_graphs,
                ent_graphs,
                etypes,
                e_labels,
                labels,
                output_kg_scores=False
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        # bs, sq = input_ids.shape
        # sequence_output = torch.zeros(bs, sq, self.config.hidden_size).to(input_ids.device)
        # attention = torch.zeros(bs, self.config.num_attention_heads, sq, sq).to(input_ids.device)
        # seq_lens = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        # for batch_i, seq_len in enumerate(seq_lens):
        #     if seq_len <= 1024:
        #         output, attn = self.encode(input_ids[batch_i, :seq_len].unsqueeze(0), attention_mask[batch_i, :seq_len].unsqueeze(0))
        #         sequence_output[batch_i, :seq_len, :] = output.squeeze(0)
        #         attention[batch_i, :, :seq_len, :seq_len] = attn.squeeze(0)
        #     elif seq_len <= 2048:
        #         input_ids1 = input_ids[batch_i, :1024].unsqueeze(0)
        #         input_ids2 = input_ids[batch_i, 1024: seq_len].unsqueeze(0)
        #         attention_mask1 = attention_mask[batch_i, :1024].unsqueeze(0)
        #         attention_mask2 = attention_mask[batch_i, 1024: seq_len].unsqueeze(0)
        #         output1, attn1 = self.encode(input_ids1, attention_mask1)
        #         output2, attn2 = self.encode(input_ids2, attention_mask2)
        #         output = torch.cat([output1.squeeze(0), output2.squeeze(0)])
        #         sequence_output[batch_i, :seq_len, :] = output
        #         attention[batch_i, :, :1024, :1024] = attn1.squeeze(0)
        #         attention[batch_i, :, 1024:seq_len, 1024:seq_len] = attn2.squeeze(0)
        #     else:
        #         input_ids1 = input_ids[batch_i, :1024].unsqueeze(0)
        #         input_ids2 = input_ids[batch_i, 1024: 2048].unsqueeze(0)
        #         input_ids3 = input_ids[batch_i, 2048: seq_len].unsqueeze(0)
        #         attention_mask1 = attention_mask[batch_i, :1024].unsqueeze(0)
        #         attention_mask2 = attention_mask[batch_i, 1024: 2048].unsqueeze(0)
        #         attention_mask3 = attention_mask[batch_i, 2048: seq_len].unsqueeze(0)
        #         output1, attn1 = self.encode(input_ids1, attention_mask1)
        #         output2, attn2 = self.encode(input_ids2, attention_mask2)
        #         output3, attn3 = self.encode(input_ids3, attention_mask3)
        #         output = torch.cat([output1.squeeze(0), output2.squeeze(0), output3.squeeze(0)])
        #         sequence_output[batch_i, :seq_len, :] = output
        #         attention[batch_i, :, :1024, :1024] = attn1.squeeze(0)
        #         attention[batch_i, :, 1024:2048, 1024:2048] = attn2.squeeze(0)
        #         attention[batch_i, :, 2048:seq_len, 2048:seq_len] = attn3.squeeze(0)
        batch_size, num_heads, seq_len, seq_len = attention.size()
        sequence_output[:, self.max_seq_length:, :] = 0
        device = sequence_output.device
        nes = [len(x) for x in entity_pos]
        ne = max(nes)
        nss = [len(x) for x in sent_pos]
        hs_e, rs_e, ts_e, e_scores, kg_flag = self.get_hrt(sequence_output, attention, hts, sent_pos, entity_pos, coref_pos, mention_pos, men_graphs, ent_graphs, etypes, ne)
        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))

        b1_e = hs_e.view(batch_size, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(batch_size, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(batch_size, ne, ne, self.emb_size * self.block_size)

        # if neg_mask is not None:
        #     bl_e = bl_e * neg_mask.unsqueeze(-1)

        feature = self.projection(bl_e)
        feature = self.axial_conv(feature) + feature  # (4, 42, 42, 768)
        # torch.save(feature, 'xxxxxxxx')
        # ss_e = ss.unsqueeze(1).unsqueeze(1).expand(-1, 42, 42, -1, -1)
        # feature_s = feature.unsqueeze(-2).expand(-1, -1, -1, 25, -1)
        # fs = torch.cat([feature_s, ss_e], dim=-1)  # (4, 42, 42, 25, 768 * 2)
        #
        # sent_logits = self.sent_projection(fs)  # (4, 42, 42, 25)
        rel_logits = self.classifier(feature)  # (4, 42, 42, 97)

        self_mask = (1 - torch.diag(torch.ones(ne))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        rel_logits = rel_logits * self_mask
        # sent_logits = sent_logits * sent_mask
        flat_rel_logits = torch.cat([
            rel_logits[x, :nes[x], :nes[x], :].reshape(-1, self.num_class) for x in range(batch_size)
        ])
        # flat_sent_logits = torch.cat([
        #     sent_logits[x, :nes[x], :nes[x], :].reshape(-1, 25) for x in range(batch_size)
        # ])

        if labels is None:  # 
            logits = flat_rel_logits
            if output_kg_scores:  # case study
                return self.rel_loss_fnt.get_label(logits), e_scores
            else:  # 
                return self.rel_loss_fnt.get_label(logits)
        else:  # 
            rel_loss = self.rel_loss_fnt(flat_rel_logits, labels)
            if self.kg_loss_weight < 0 or not kg_flag:
                return rel_loss
            kg_loss = self.kg_loss_fnt(e_scores, e_labels)
            output = rel_loss + self.kg_loss_weight * kg_loss
            return output
