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
                 xd=None, xt=None, labels=None, transform=None,
                 pre_transform=None, smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
#        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
#            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
#            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(smile_graph)
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

    def process(self, smile_graph):
        data_list = []
#        print(smile_graph)

        c_size, features, edge_index = smile_graph
            
        for j in range(len(c_size)):
            GCNData = DATA.Data(x = torch.Tensor(features[j]),
                                edge_index = torch.LongTensor(edge_index[j]).transpose(1, 0))                
            data_list.append(GCNData)
                
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

#        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        
        torch.save((data, slices), self.processed_paths[0])
