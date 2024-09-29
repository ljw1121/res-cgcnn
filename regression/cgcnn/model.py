from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.初始化
        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len     # 原子隐藏特征
        self.nbr_fea_len = nbr_fea_len       # 键特征
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)                # 通过线性层，将2*原子矩阵和边矩阵的维度转化为2*原子矩阵维度(ZW+b)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()                        # 激活函数
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        Parameters
        ----------
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        Returns
        -------
        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution
        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape               # N是一个batch中的原子数，M是键的个数(度矩阵)
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        # Z=vi+vj+Uij
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),       # cat()拼接函数，拼接的矩阵必须维度相同；unsqueeze(1)增加一个维度，expand在增加维度的基础上复制数据，以满足新维度
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)                        # 通过线性层，改变维度(Zw+b)
        total_gated_fea = self.bn1(total_gated_fea.view(                     # batchnorm
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)

        # 文章中的sigmoid()和g()
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)               # chunk分成两个数组
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class Res_block(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, nbr_fea_len,
                 atom_fea_len=64):
        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.  输入原子特征数
        nbr_fea_len: int
          Number of bond features.   键的特征数
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers    卷积层数
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling  线性层数
        """
        super(Res_block, self).__init__()
        self.res_conv = ConvLayer(atom_fea_len=atom_fea_len,    # 自定义卷积层,atom_fea_len=atom_fea_len=64
                                    nbr_fea_len=nbr_fea_len)                # 键的个数
                                                
        self.conv_to_fc_softplus = nn.Softplus()                            # softplus操作，非线性激活
  
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):       
        return self.conv_to_fc_softplus(self.res_conv(atom_fea, nbr_fea, nbr_fea_idx) +  atom_fea)    
    

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, n_res=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.  输入原子特征数
        nbr_fea_len: int
          Number of bond features.   键的特征数
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers    卷积层数
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling  线性层数
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)         # (输入原子特征数,原子隐藏特征数64)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,    # 自定义卷积层,atom_fea_len=atom_fea_len=64
                                    nbr_fea_len=nbr_fea_len)                # 键的个数
                                    for _ in range(n_conv)])                # 卷积了n_conv=3层
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)                # 线性层，(原子隐藏特征数64,池化后隐藏特征数)

        self.res_blocks = nn.ModuleList([Res_block(atom_fea_len=atom_fea_len,nbr_fea_len=nbr_fea_len)
                                        for _ in range(n_res)])

        self.conv_to_fc_softplus = nn.Softplus()                            # softplus操作，非线性激活
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features
        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]                     # 规范化求和作为池函数
        return torch.cat(summed_fea, dim=0)
    

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch 晶体个数
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type   原子类型中获得的原子特征
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx  从晶体idx映射到原子idx
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)                             # 这是一个线性层，(输入原子特征数,原子隐藏特征数64)
 
        for conv_func in self.convs:
            atom_fea1 = conv_func(atom_fea, nbr_fea, nbr_fea_idx) 
        for res_block in self.res_blocks:
            atom_fea2 = res_block(atom_fea1, nbr_fea, nbr_fea_idx)

  
        crys_fea = self.pooling(atom_fea2, crystal_atom_idx)             # 池化
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))  # 隐藏线性层，(原子隐藏特征数,池化后隐藏特征数)
        crys_fea = self.conv_to_fc_softplus(crys_fea)                   # softplus操作
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out




