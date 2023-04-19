from utils import *
from trainer import *
from neg_sampler import *
from load_model import *
from splitter import *

def get_data(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    if name == 'mtl':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = mtl_data(path, args)
        if args.mtl_task_num == 2:
            train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
        else:
            train_dataset = (train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
            val_dataset = (val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values)
            test_dataset = (test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values)
        train_dataset = mtlDataSet(train_dataset, args)
        val_dataset = mtlDataSet(val_dataset, args)
        test_dataset = mtlDataSet(test_dataset, args)

        # dataloader
        train_dataloader = get_train_loader(train_dataset, args)
        val_dataloader = get_val_loader(val_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)

        return train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict
    
    elif name == 'sequence':
        _, data, user_count, item_count = sequencedataset(args.item_min, args, path)
        args.num_users = user_count
        args.num_items = item_count
        train_data, val_data, test_data = train_val_test_split(data)
        train_data_s, val_data_s = {}, {}
        data_len = len(train_data)
        i = 0
        for key, _ in val_data.items():
            train_data_s[key] = train_data[key]
            val_data_s[key] = val_data[key]
            i += 1
            if i == int(data_len / args.valid_rate):
                break
        if 'bert' in args.model_name:
            train_dataset = BertTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)#
        else:
            train_dataset = BuildTrainDataset(train_data, args.max_len, args.bert_mask_prob, args.pad_token, args.num_items, rng)
        valid_dataset = Build_full_EvalDataset(train_data_s, val_data_s, args.max_len, args.pad_token, args.num_items)
        test_dataset = Build_full_EvalDataset(train_data, test_data, args.max_len, args.pad_token, args.num_items)
        train_dataloader = get_train_loader(train_dataset, args)
        valid_dataloader = get_val_loader(valid_dataset, args)
    else:
        raise ValueError('unknown dataset name: ' + name)