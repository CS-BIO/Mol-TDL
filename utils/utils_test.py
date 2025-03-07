import os
from itertools import islice
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def process(self, xd, y, smile_graph):
        assert (len(xd) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_list2 = []
        data_list3 = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            labels = y[i]
            c_size, features, edge_index = smile_graph[xd[i]]
#            print("========================================")
#            print(len(c_size))
#            print(len(features))
#            print(len(edge_index))
#            print(len(edge_feat))
#            print(len(edge_edge_index))
            
            for j in range(len(c_size)):
                GCNData = DATA.Data(x = torch.Tensor(features[j]),
                                    edge_index = torch.LongTensor(edge_index[j]).transpose(1, 0),
                                    y=torch.Tensor([labels]))
#                GCNData2 = DATA.Data(x = torch.Tensor(edge_feat[j]),
#                                    edge_index = torch.LongTensor(edge_edge_index[j]).transpose(1, 0),
#                                    y=torch.Tensor([labels]))
#                GCNData3 = DATA.Data(x = torch.Tensor(tri_feat[j]),
#                                     edge_index = torch.LongTensor(tri_tri_index[j]).transpose(1, 0),
#                                     y=torch.Tensor([labels]))
                
#                GCNData = DATA.Data(x = torch.Tensor(edge_feat[j]),
#                                    edge_index = torch.LongTensor(edge_edge_index[j]).transpose(1, 0),
#                                    y=torch.Tensor([labels]))
                
#                GCNData.__setitem__('edge_size', torch.Tensor([edge_size[j]]))
#                GCNData.__setitem__('edge_feat', torch.Tensor(edge_feat[j]))
#                GCNData.__setitem__('edge_edge_index',  torch.LongTensor(edge_edge_index[j]).transpose(1, 0))
                
#                GCNData.__setitem__('edge_size', torch.Tensor([tri_size[j]]))
#                GCNData.__setitem__('edge_feat', torch.Tensor(tri_feat[j]))
#                GCNData.__setitem__('edge_edge_index',  torch.LongTensor(tri_tri_index[j]).transpose(1, 0))                
                
                data_list.append(GCNData)
#                data_list2.append(GCNData2)
#                data_list3.append(GCNData3)
                
                
#                GCNData = DATA.Data(x = torch.Tensor(edge_feat[j]),
#                                    edge_index = torch.LongTensor(edge_edge_index[j]).transpose(1, 0),
#                                    y=torch.Tensor([labels]))
#                data_list.append(GCNData)
                
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
#        print(len(data_list))
#        print(data_list[0])
#        print(data_list[0].y)
#        print(data_list[0].edge_x)
#        print(data_list[0].edge_edge)
        data, slices = self.collate(data_list)
#        data2, slices2 = self.collate(data_list2)
#        data3, slices3 = self.collate(data_list3)
        # save preprocessed data:
#        print(self.processed_paths)
        torch.save((data, slices), self.processed_paths[0])
#        torch.save((data2, slices2), self.processed_paths[0]+"2")
#        torch.save((data3, slices3), self.processed_paths[0]+"3")
