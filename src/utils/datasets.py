from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit

class GDSet(Dataset):
    def __init__(self, g):
        self.g = g

    def __getitem__(self, index):

        g, label = self.g[index]
        for i in range(len(g)):
          g[i]['posi_ids'] = i
        return g, label

    def __len__(self):
        return len(self.g)
    

def split_dataset(dataset, train_params, random_seed=1):
    rs = ShuffleSplit(n_splits=1, test_size=train_params.get('test_split'), random_state=random_seed)
    for i, (train_index_tmp, test_index) in enumerate(rs.split(dataset)):
        rs2 = ShuffleSplit(n_splits=1, test_size=train_params.get('val_split'), random_state=random_seed)
        for j, (train_index, val_index) in enumerate(rs2.split(train_index_tmp)):
            train_index = train_index_tmp[train_index]
            val_index = train_index_tmp[val_index]
            trainDSet = [dataset[x] for x in train_index]
            valDSet = [dataset[x] for x in val_index]
            testDSet = [dataset[x] for x in test_index]
    return trainDSet, valDSet, testDSet