import torch.nn as nn
init = nn.init.xavier_uniform_
import torch as t
class GraphTransformerLayer(nn.Module):
    def __init__(self, config):
        super(GraphTransformerLayer, self).__init__()
        self.config = config
        self.qTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))
        self.kTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))
        self.vTrans = nn.Parameter(init(t.empty(config['embedding_size'], config['embedding_size'])))

    def forward(self, adj, embeds, flag=False):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.config['K'], self.config['embedding_size'] // self.config['K']])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = embeds.new_zeros([adj.shape[0], self.config['K']])
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.config['embedding_size']])
        tem = embeds.new_zeros([adj.shape[0], self.config['embedding_size']])
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds, att
