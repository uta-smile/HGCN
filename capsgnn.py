"""CapsGNN Trainer."""
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.autograd import Variable
from denseGCNConv import DenseGCNConv
from torch.nn import Linear, BatchNorm1d
from layers import SecondaryCapsuleLayer, firstCapsuleLayer, ReconstructionNet

class CapsGNN(torch.nn.Module):
    def __init__(self, args, number_of_features, number_of_targets, max_node_num):
        super(CapsGNN, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self.max_node_num = max_node_num
        self._setup_layers()
    
    def _setup_firstCapsuleLayer(self):
        self.first_capsule = firstCapsuleLayer(number_of_features=self.number_of_features, 
                                               max_node_num=self.max_node_num, 
                                               capsule_dimensions=self.args.capsule_dimensions,
                                               disentangle_num=self.args.disentangle_num, 
                                               dropout=self.args.dropout)

    def _setup_hidden_capsules(self):
        self.hidden_capsule = SecondaryCapsuleLayer(num_iterations=self.args.num_iterations,
                                                   num_routes=self.max_node_num,
                                                   num_capsules=self.args.capsule_num,
                                                   in_channels=self.args.capsule_dimensions,
                                                   out_channels=self.args.capsule_dimensions,
                                                   dropout=self.args.dropout)
                                                               
    def _setup_class_capsule(self):
        self.class_capsule = SecondaryCapsuleLayer(num_iterations=self.args.num_iterations,
                                                   num_routes=self.args.capsule_num,
                                                   num_capsules=self.number_of_targets,
                                                   in_channels=self.args.capsule_dimensions,
                                                   out_channels=self.args.capsule_dimensions,
                                                   dropout=self.args.dropout)

    def _setup_reconstructNet(self):
        self.recons_net = ReconstructionNet(n_dim=self.args.capsule_dimensions, 
                                            n_classes=self.number_of_targets, 
                                            hidden=self.args.capsule_dimensions)

    def _setup_layers(self):
        self._setup_firstCapsuleLayer()
        self._setup_hidden_capsules()
        self._setup_class_capsule()
        self._setup_reconstructNet()
        
    def cal_recons_loss(self, pred_adj, adj, mask=None):
        eps = 1e-7
        # Each entry in pred_adj cannot larger than 1
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
        # The diagonal entries in pred_adj should be 0
        pred_adj = pred_adj.masked_fill_(torch.eye(adj.size(1), adj.size(1)).bool().to('cuda'), 0)
        # Cross entropy loss
        link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)

        if mask is not None:
            num_entries = torch.sum(torch.sum(mask, dim=1) ** 2)
            adj_mask = mask.unsqueeze(2).float() @ torch.transpose(mask.unsqueeze(2).float(), 1, 2)
            link_loss[(1-adj_mask).bool()] = 0.0
        else:
            num_entries = pred_adj.size(0) * pred_adj.size(1) * pred_adj.size(2)

        link_loss = torch.sum(link_loss) / float(num_entries)
        return link_loss

    def forward(self, x, adj_in, mask, batch, y):
        batch_size = x.size(0)
        out = self.first_capsule(x, adj_in, mask, batch)
        residual = out
        out, c_ij, adj = self.hidden_capsule(out, adj_in, mask)
        out = out.view(batch_size, -1, self.args.capsule_dimensions)
        
        adj = torch.min(adj, torch.ones(1, dtype=adj.dtype).cuda())
        adj = adj.masked_fill_(torch.eye(adj.size(1), adj.size(1)).bool().to('cuda'), 0)
        
        out, c_ij, adj = self.class_capsule(out, adj)
        out = out.squeeze(4).squeeze(1)
        recons_out = self.recons_net(residual, out, y)  # reconstructed adjacency matrix
        recon_loss = self.cal_recons_loss(recons_out, adj_in, mask)
        
        out = (torch.sqrt((out ** 2).sum(2))).view(batch_size, self.number_of_targets)
        return out, recon_loss