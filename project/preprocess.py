import pandas as pd
from tqdm import tqdm
from ast import literal_eval


def split_dataset(dataset_path):

    print('----------load dataset...----------')

    #---------------read_dataset--------------#
    all_dataset = pd.read_csv(dataset_path, sep=',', header=None, names=['user', 'traj', 'time', 'category'])
    
    #-------------statistical corpus----------#
    user_list = all_dataset['user'].drop_duplicates().values.tolist()
    poi_list = set()
    category_list = set()
    
    for idx, (_, traj, _, category) in tqdm(all_dataset.iterrows(), total=len(all_dataset), ncols=100):
        poi_list.update(literal_eval(traj))
        category_list.update(literal_eval(category))
    
    poi_nums = len(poi_list)
    category_nums = len(category_list)
    user_nums = len(user_list)
    
    #--------split train-test dataset--------#
    train_nums = 0
    user_traj_train, user_traj_test = {}, {}
    for user in tqdm(user_list, ncols=100):
        user_traj_train[user], user_traj_test[user] = [], []
        one_user_data = all_dataset.loc[all_dataset.user==user,:]
        one_user_data_train = one_user_data.iloc[:int(0.8*len((one_user_data)))]
        one_user_data_test = one_user_data.iloc[int(0.8*len((one_user_data))):]

        for idx , (_, one_row) in enumerate(one_user_data_train.iterrows()):
            train_nums += 1
            _, traj, time, category = one_row
            user_traj_train[user].append((literal_eval(traj), literal_eval(time), literal_eval(category), idx))
        
        for idx, (_, one_row) in enumerate(one_user_data_test.iterrows()):
            _, traj, time, category = one_row
            user_traj_test[user].append((literal_eval(traj), literal_eval(time), literal_eval(category), idx))
    
    print('-------Finish loading data!--------')
    return user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums


if __name__=='__main__':
    user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums = split_dataset('./data/weeplace/weeplace_all.csv')
    print(user_traj_test)
    print(train_nums, poi_nums, category_nums , user_nums)