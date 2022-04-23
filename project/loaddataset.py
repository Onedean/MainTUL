import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from preprocess import split_dataset


class TulDataset(Dataset):
    def __init__(self, data):
        self.poi_seq, self.time_seq, self.category_seq, self.idx_seq, self.input_len, self.label = [], [], [], [], [], []
        for user_id in data:
            for one_traj in data[user_id]:
                self.poi_seq.append(one_traj[0])
                self.time_seq.append(one_traj[1])
                self.category_seq.append(one_traj[2])
                self.idx_seq.append(one_traj[3])
                self.input_len.append(len(one_traj[0]))
                self.label.append(user_id)
    
    def __getitem__(self, index):
        return self.poi_seq[index], self.time_seq[index], self.category_seq[index], self.idx_seq[index], self.input_len[index], self.label[index]
    
    def __len__(self):
        return len(self.poi_seq)


class TulCollator(object):
    def __init__(self, long_term_num=8, user_traj_train=None):
        self.long_term_num = long_term_num
        self.user_traj_train = user_traj_train
    
    def long_term_augment(self, one_batch_idx, one_batch_label):
        longterm_poi_seq = []
        longterm_category_seq = []
        longterm_hour_seq = []
        longterm_time_seq = []
        longterm_len = []
        max_len = 0
        for idx, user_id in enumerate(one_batch_label):
            
            # start_idx, end_idx = one_batch_idx[idx] - int(self.long_term_num / 2), one_batch_idx[idx] + int(self.long_term_num / 2) + 1
            # if start_idx < 0:
            #     start_idx = 0
            # if end_idx > len(self.user_traj_train[user_id]):
            #     end_idx = len(self.user_traj_train[user_id])
            # long_term_idx = range(start_idx, end_idx)
            #start_idx = 0 if one_batch_idx[idx] - self.long_term_num < 0 else one_batch_idx[idx] - self.long_term_num
            #long_term_idx = range(start_idx, one_batch_idx[idx]+1)
            
            long_term_idx = sorted(set(random.sample(range(0, len(self.user_traj_train[user_id])), self.long_term_num) + [one_batch_idx[idx]]))
            
            traj_seq, time_seq, category_seq = [], [], []
            for one_traj_idx in long_term_idx:
                traj_seq.extend(self.user_traj_train[user_id][one_traj_idx][0])
                time_seq.extend(self.user_traj_train[user_id][one_traj_idx][1])
                category_seq.extend(self.user_traj_train[user_id][one_traj_idx][2])
            max_len = max(max_len, len(traj_seq))
            longterm_poi_seq.append(traj_seq)
            longterm_category_seq.append(category_seq)
            longterm_hour_seq.append((((np.array(time_seq) % (24 * 60 * 60) / 60 / 60) + 8) % 24 + 1).tolist())
            longterm_time_seq.append(time_seq)
            longterm_len.append(len(traj_seq))
        longterm_poi_seq = [traj + [0] * (max_len - len(traj)) for traj in longterm_poi_seq]
        longterm_category_seq = [traj + [0] * (max_len - len(traj)) for traj in longterm_category_seq]
        longterm_hour_seq = [hour + [0] * (max_len - len(hour)) for hour in longterm_hour_seq]
        longterm_time_seq = [time + [0] * (max_len - len(time)) for time in longterm_time_seq]
        return longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len
    

    def collate_train(self, batch):
        batch = sorted(batch, key=lambda x:x[-2], reverse=True)
        one_batch_poi, one_batch_time, one_batch_category, one_batch_idx, one_batch_len, one_batch_label = zip(*batch)
        one_batch_maxlen = max(one_batch_len)
        
        current_poi_seq = [one_poi_seq + [0] * (one_batch_maxlen - len(one_poi_seq)) for one_poi_seq in one_batch_poi]
        current_category_seq = [one_category_seq + [0] * (one_batch_maxlen - len(one_category_seq)) for one_category_seq in one_batch_category]
        current_hour_seq = [(((np.array(one_time_seq) % (24 * 60 * 60) / 60 / 60) + 8) % 24 + 1).tolist() + [0] * (one_batch_maxlen - len(one_time_seq)) for one_time_seq in one_batch_time]
        current_time_seq = [one_time_seq + [0] * (one_batch_maxlen - len(one_time_seq)) for one_time_seq in one_batch_time]
        current_len = one_batch_len

        longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len = self.long_term_augment(one_batch_idx, one_batch_label)

        return torch.LongTensor(current_poi_seq), torch.LongTensor(current_category_seq), torch.LongTensor(current_hour_seq), torch.LongTensor(current_time_seq), torch.LongTensor(current_len), torch.LongTensor(longterm_poi_seq), torch.LongTensor(longterm_category_seq), torch.LongTensor(longterm_hour_seq), torch.LongTensor(longterm_time_seq), torch.LongTensor(longterm_len), torch.LongTensor(one_batch_label)
    

    def collate_test(self, batch):
        batch = sorted(batch, key=lambda x:x[-2], reverse=True)
        one_batch_poi, one_batch_time, one_batch_category, _, one_batch_len, one_batch_label = zip(*batch)
        one_batch_maxlen = max(one_batch_len)
        
        current_poi_seq = [one_poi_seq + [0] * (one_batch_maxlen - len(one_poi_seq)) for one_poi_seq in one_batch_poi]
        current_category_seq = [one_category_seq + [0] * (one_batch_maxlen - len(one_category_seq)) for one_category_seq in one_batch_category]
        current_hour_seq = [(((np.array(one_time_seq) % (24 * 60 * 60) / 60 / 60) + 8) % 24 + 1).tolist() + [0] * (one_batch_maxlen - len(one_time_seq)) for one_time_seq in one_batch_time]
        current_time_seq = [one_time_seq + [0] * (one_batch_maxlen - len(one_time_seq)) for one_time_seq in one_batch_time]
        current_len = one_batch_len
        
        return torch.LongTensor(current_poi_seq), torch.LongTensor(current_category_seq), torch.LongTensor(current_hour_seq), torch.LongTensor(current_time_seq), torch.LongTensor(current_len), torch.LongTensor(one_batch_label)
        

def get_dataloader(traj_dataset, load_datatype, batch_size, sampler=None, long_term_num=None, user_traj_train=None):
    if load_datatype=='train':
        traj_dataloader = DataLoader(traj_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False, collate_fn=TulCollator(long_term_num, user_traj_train).collate_train)
    elif load_datatype=='valid':
        traj_dataloader = DataLoader(traj_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False, collate_fn=TulCollator(long_term_num, user_traj_train).collate_train)
    elif load_datatype=='test':
        traj_dataloader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=TulCollator().collate_test)
    else:
        raise Exception("Unkonwn data type!")
    return traj_dataloader


def get_dataset(user_traj_train, user_traj_test, train_nums):
    indices = list(range(train_nums))
    split = int(np.floor(train_nums * 0.8))
    np.random.seed(666)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_dataset = TulDataset(data=user_traj_train)
    test_dataset = TulDataset(data=user_traj_test)
    return train_dataset, test_dataset, train_sampler, valid_sampler


if __name__=='__main__':
    user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums = split_dataset('./data/weeplace/weeplace_mini.csv')
    train_dataset, test_dataset, train_sampler, valid_sampler = get_dataset(user_traj_train, user_traj_test, train_nums)
    train_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='train', batch_size=3, long_term_num=4, sampler=train_sampler, user_traj_train=user_traj_train)
    for current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len, longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len, one_batch_label  in train_dataloader:
        print(current_poi_seq.shape)
        print(current_poi_seq)
        print(current_category_seq.shape)
        print(current_category_seq)
        print(current_hour_seq.shape)
        print(current_hour_seq)
        print(current_time_seq.shape)
        print(current_time_seq)
        print(current_len)
        print('-'*10)
        print(longterm_poi_seq.shape)
        print(longterm_poi_seq)
        print(longterm_category_seq.shape)
        print(longterm_category_seq)
        print(longterm_hour_seq.shape)
        print(longterm_hour_seq)
        print(longterm_time_seq.shape)
        print(longterm_time_seq)
        print(longterm_len.shape)
        print(longterm_len)
        print('-'*10)
        print(one_batch_label.shape)
        print(one_batch_label)
        exit()