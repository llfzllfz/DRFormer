import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools.config import get_config
from tools.utils import set_seed
from sklearn.model_selection import KFold

def read_tsv(path):
    df = pd.read_csv(path, sep='\t', header=None, names = ['Type', 'loc', 'Seq', 'Str', 'Score', 'label'])
    df = df.loc[df['Type'] != "Type"]
    def get_type_loc(item):
        return '{}|{}'.format(item['Type'], item['loc'])
    df['Type_loc'] = df.apply(get_type_loc, axis = 1)
    Type_loc = df['Type_loc'].to_numpy()
    sequences = df['Seq'].to_numpy()
    structs  = df['Str'].to_numpy()
    targets   = df['Score'].to_numpy().astype(np.float32).reshape(-1)
    targets[targets<0] = 0
    targets[targets>0] = 1
    return sequences, structs, targets, Type_loc

if __name__ == '__main__':
    config = get_config()
    print(config)
    assert config.command == 'clip', 'Please choose true mode of config'
    set_seed(config.seed)

    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    if not os.path.exists(config.k_fold_data_path):
        os.mkdir(config.k_fold_data_path)
    for fold in folds:
        if not os.path.exists(os.path.join(config.k_fold_data_path, fold)):
            os.mkdir(os.path.join(config.k_fold_data_path, fold))

    root = config.origin_data_path

    file_list = os.path.join(config.origin_data_path, 'all.list')
    with open(file_list, 'r') as f:
        tmp = f.readlines()
    f.close()

    filenames = [_.replace('\n', '') for _ in tmp]
    tl_data = pd.DataFrame({})
    for filename in tqdm(filenames):
        path = os.path.join(root, filename + '.tsv')
        sequences, structs, targets, Type_loc = read_tsv(path)
        tmp = pd.DataFrame({'Type_loc': Type_loc, 'Seq': sequences, 'Str':structs, 'Label':targets})
        tmp_0 = tmp[tmp['Label'] == 0].reset_index()
        tl_tmp_0 = tmp_0.sample(frac = config.tl_ratio, random_state = config.seed)
        tl_tmp_0['filename'] = filename
        k_fold_tmp_0 = tmp_0.drop(tl_tmp_0.index)
        tl_data = pd.concat([tl_data, tl_tmp_0])

        tmp_1 = tmp[tmp['Label'] == 1].reset_index()
        tl_tmp_1 = tmp_1.sample(frac = config.tl_ratio, random_state = config.seed)
        tl_tmp_1['filename'] = filename
        k_fold_tmp_1 = tmp_1.drop(tl_tmp_1.index)
        tl_data = pd.concat([tl_data, tl_tmp_1])
        
        kf = KFold(n_splits = 5, shuffle = True, random_state = config.seed)
        result = []
        for train_index, test_index in kf.split(k_fold_tmp_0):
            result.append(k_fold_tmp_0.iloc[test_index])
        for idx, (train_index, test_index) in enumerate(kf.split(k_fold_tmp_1)):
            result_tmp = pd.concat([result[idx], k_fold_tmp_1.iloc[test_index]])
            result_tmp.to_csv(os.path.join(os.path.join(config.k_fold_data_path, 'fold_{}'.format(idx+1)), filename + '.csv'), index=False)
    tl_data.to_csv(os.path.join(config.tl_data_path, 'clip_tl.csv'), index=False)
