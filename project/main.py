import time
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from loaddataset import get_dataset, get_dataloader
from preprocess import split_dataset
from utils import EarlyStopping, accuracy_1, accuracy_5, loss_with_plot
from sklearn.metrics import f1_score, precision_score, recall_score
from model import StudentEncoder, TeacherEncoder, TulNet, PositionalEncoding, TemporalEncoding, TransformerTimeAwareEmbedding, LstmTimeAwareEmbedding
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='MainTUL')
    parse.add_argument('--times', type=int, default=1, help='times of repeat experiment')
    parse.add_argument('--dataset', type=str, default="foursquare_mini", help='dataset for experiment')
    parse.add_argument('--encoding_type', type=str, default="temporal", help='time aware embedding strategy')
    parse.add_argument('--epochs', type=int, default=30, help='Number of total epochs')
    parse.add_argument('--batch_size', type=int, default=512, help='Size of one batch')
    parse.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parse.add_argument('--temperature', type=float, default=10, help='Temperature hyperparameter')
    parse.add_argument('--lambda_parm', type=float, default=10, help='Distillation loss hyperparameter')
    parse.add_argument('--long_term_num', type=int, default=8, help='the number of long term augment day')
    parse.add_argument('--embed_size', type=int, default=512, help='Number of embeding dim')
    parse.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parse.add_argument('--num_layers', type=int, default=3, help='Number of EncoderLayer')

    args = parse.parse_args()
    return args


def getLogger(dataset):
    """[Define logging functions]

    Args:
        dataset ([string]): [dataset name]
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename='./project/log/'+dataset+'.log', mode='w')
    consoleHandler.setLevel(logging.INFO)

    consoleformatter = logging.Formatter("%(message)s")
    fileformatter = logging.Formatter("%(message)s")

    consoleHandler.setFormatter(consoleformatter)
    fileHandler.setFormatter(fileformatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

def compute_loss_student_ce(student_output, target):
    return F.cross_entropy(student_output, target)

def compute_loss_teacher_ce(teacher_output, target):
    return F.cross_entropy(teacher_output, target)

def compute_loss_dis(student_output, teacher_output, temperature):
    prob_student = F.log_softmax(student_output / temperature, dim=1)
    prob_teacher = F.softmax(teacher_output / temperature, dim=1)
    return F.kl_div(prob_student, prob_teacher, size_average=False) * (temperature**2) / student_output.shape[0]


def train_model(train_dataset, train_sampler, valid_sampler, model, optimizer, user_traj_train, devices, args, logger):
    
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(logger=logger, dataset_name=args.dataset, patience=3, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.9)

    for epoch_idx in range(args.epochs):

        model.train()
        train_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='train', batch_size=args.batch_size, sampler=train_sampler, long_term_num=args.long_term_num, user_traj_train=user_traj_train)
        loss_train_list = []
        
        for batch_idx, (current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len, longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len, one_batch_label) in enumerate(tqdm(train_dataloader)):

            current_poi_seq, current_category_seq, current_hour_seq, current_time_seq = current_poi_seq.to(devices[0]), current_category_seq.to(devices[0]), current_hour_seq.to(devices[0]), current_time_seq.to(devices[0])
            longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq = longterm_poi_seq.to(devices[0]), longterm_category_seq.to(devices[0]), longterm_hour_seq.to(devices[0]), longterm_time_seq.to(devices[0])
            current_len, longterm_len, one_batch_label = current_len.to(devices[0]), longterm_len.to(devices[0]), one_batch_label.to(devices[0])
            
            student_output, teacher_output = model(current_poi_seq, current_category_seq, current_hour_seq, current_len, longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len)
            
            loss_stu_ce_1 = compute_loss_student_ce(student_output, one_batch_label)
            loss_tea_ce_1 = compute_loss_teacher_ce(teacher_output, one_batch_label)
            loss_ce_1 = loss_stu_ce_1 + loss_tea_ce_1
            
            loss_dis_1 = compute_loss_dis(student_output, teacher_output, temperature=args.temperature)
            loss_sum_1 = loss_ce_1 + args.lambda_parm * loss_dis_1

            student_output, teacher_output = model(longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_len, current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len)

            loss_stu_ce_2 = compute_loss_student_ce(student_output, one_batch_label)
            loss_tea_ce_2 = compute_loss_teacher_ce(teacher_output, one_batch_label)
            loss_ce_2 = loss_stu_ce_2 + loss_tea_ce_2
            
            loss_dis_2 = compute_loss_dis(teacher_output, student_output, temperature=args.temperature)
            loss_sum_2 = loss_ce_2 + args.lambda_parm * loss_dis_2
            
            loss_sum = loss_sum_1 + loss_sum_2

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            loss_train_list.append(loss_sum.item())
            
            if not(batch_idx % 40):
                output_content = "Train epoch:{} batch:{} loss_stu_ce:{:.6f} loss_tea_ce:{:.6f} loss_dis:{:.6f} loss_sum:{:.6f} loss:{:.6f} acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f} macro_f1:{:.6f}"
                logger.info(output_content.format(epoch_idx, batch_idx, loss_stu_ce_1.item(), loss_tea_ce_1.item(), loss_dis_1.item(), loss_sum_1.item(), np.mean(loss_train_list)))
                logger.info(output_content.format(epoch_idx, batch_idx, loss_stu_ce_2.item(), loss_tea_ce_2.item(), loss_dis_2.item(), loss_sum_2.item(), np.mean(loss_train_list)))
        

        model.eval()
        valid_dataloader = get_dataloader(traj_dataset = train_dataset, load_datatype='valid', batch_size=args.batch_size, sampler=valid_sampler, long_term_num=args.long_term_num, user_traj_train=user_traj_train)
        loss_valid_list, y_predict_list, y_true_list, acc1_list, acc5_list = [], [], [], [], []

        with torch.no_grad():

            for batch_idx, (current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len, longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len, one_batch_label) in enumerate(tqdm(valid_dataloader)):
                
                current_poi_seq, current_category_seq, current_hour_seq, current_time_seq = current_poi_seq.to(devices[0]), current_category_seq.to(devices[0]), current_hour_seq.to(devices[0]), current_time_seq.to(devices[0])
                longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq = longterm_poi_seq.to(devices[0]), longterm_category_seq.to(devices[0]), longterm_hour_seq.to(devices[0]), longterm_time_seq.to(devices[0])
                current_len, longterm_len, one_batch_label = current_len.to(devices[0]), longterm_len.to(devices[0]), one_batch_label.to(devices[0])

                student_output, teacher_output = model(current_poi_seq, current_category_seq, current_hour_seq, current_len, longterm_poi_seq, longterm_category_seq, longterm_hour_seq, longterm_time_seq, longterm_len)
                
                y_predict_list.extend(torch.max(student_output, 1)[1].cpu().numpy().tolist())
                y_true_list.extend(one_batch_label.cpu().numpy().tolist())
                acc1_list.extend(accuracy_1(student_output, one_batch_label).cpu().numpy())
                acc5_list.extend(accuracy_5(student_output, one_batch_label).cpu().numpy())

                loss_stu_ce = compute_loss_student_ce(student_output, one_batch_label)
                loss_tea_ce = compute_loss_teacher_ce(teacher_output, one_batch_label)
                loss_ce = loss_stu_ce + loss_tea_ce
                
                loss_dis = compute_loss_dis(student_output, teacher_output, temperature=args.temperature)
                loss_sum = loss_ce + args.lambda_parm * loss_dis
                loss_valid_list.append(loss_sum.item())

            macro_p = precision_score(y_true_list, y_predict_list, average='macro')
            macro_r = recall_score(y_true_list, y_predict_list, average='macro')
            macro_f1 = f1_score( y_true_list, y_predict_list, average='macro')

            output_content = "Valid epoch:{} loss:{:.6f} acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f} macro_f1:{:.6f}"
            logger.info(output_content.format(epoch_idx, np.mean(loss_valid_list), np.mean(acc1_list), np.mean(acc5_list), macro_p, macro_r, macro_f1))
        
        avg_train_losses.append(np.mean(loss_train_list))
        avg_valid_losses.append(np.mean(loss_valid_list))
        
        early_stopping(avg_valid_losses[-1], model)
        if early_stopping.early_stop:
            logger.info('Early Stop!')
            break
        else:
            scheduler.step()
    
    return avg_train_losses, avg_valid_losses


def test_model(test_dataset, model, devices, args, logger):
    model.eval()
    test_dataloader = get_dataloader(traj_dataset = test_dataset, load_datatype='test', batch_size=args.batch_size)
    loss_test_list_1, y_predict_list_1, y_true_list_1, acc1_list_1, acc5_list_1 = [], [], [], [], []
    loss_test_list_2, y_predict_list_2, y_true_list_2, acc1_list_2, acc5_list_2 = [], [], [], [], []
    with torch.no_grad():
        for current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len, one_batch_label in test_dataloader:
            
            current_poi_seq, current_category_seq, current_hour_seq = current_poi_seq.to(devices[0]), current_category_seq.to(devices[0]), current_hour_seq.to(devices[0])
            current_len, one_batch_label = current_len.to(devices[0]), one_batch_label.to(devices[0])

            output_1 = model(current_poi_seq, current_category_seq, current_hour_seq, current_len, train=False, type='1')
            
            y_predict_list_1.extend(torch.max(output_1, 1)[1].cpu().numpy().tolist())
            y_true_list_1.extend(one_batch_label.cpu().numpy().tolist())
            acc1_list_1.extend(accuracy_1(output_1, one_batch_label).cpu().numpy())
            acc5_list_1.extend(accuracy_5(output_1, one_batch_label).cpu().numpy())

            loss_1 = compute_loss_student_ce(output_1, one_batch_label)
            loss_test_list_1.append(loss_1.item())

            current_time_seq = current_time_seq.to(devices[0])

            output_2 = model(None, None, None, None, current_poi_seq, current_category_seq, current_hour_seq, current_time_seq, current_len, train=False, type='2')
            
            y_predict_list_2.extend(torch.max(output_2, 1)[1].cpu().numpy().tolist())
            y_true_list_2.extend(one_batch_label.cpu().numpy().tolist())
            acc1_list_2.extend(accuracy_1(output_2, one_batch_label).cpu().numpy())
            acc5_list_2.extend(accuracy_5(output_2, one_batch_label).cpu().numpy())

            loss_2 = compute_loss_student_ce(output_2, one_batch_label)
            loss_test_list_2.append(loss_2.item())
        
        macro_p_1 = precision_score(y_true_list_1, y_predict_list_1, average='macro')
        macro_r_1 = recall_score(y_true_list_1, y_predict_list_1, average='macro')
        macro_f1_1 = f1_score(y_true_list_1, y_predict_list_1, average='macro')

        macro_p_2 = precision_score(y_true_list_2, y_predict_list_2, average='macro')
        macro_r_2 = recall_score(y_true_list_2, y_predict_list_2, average='macro')
        macro_f1_2 = f1_score(y_true_list_2, y_predict_list_2, average='macro')

        output_content = "Test \t loss:{:.6f} acc@1:{:.6f} acc@5:{:.6f} macro_p:{:.6f} macro_r:{:.6f} macro_f1:{:.6f}"
        logger.info(output_content.format(np.mean(loss_test_list_1), np.mean(acc1_list_1), np.mean(acc5_list_1), macro_p_1, macro_r_1, macro_f1_1))
        logger.info(output_content.format(np.mean(loss_test_list_2), np.mean(acc1_list_2), np.mean(acc5_list_2), macro_p_2, macro_r_2, macro_f1_2))



def main():
    args = parse_args()
    logger = getLogger(args.dataset)
    dataset_path = './data/'+ args.dataset + '.csv'

    #---------------------dataset split----------------#
    user_traj_train, user_traj_test, train_nums, poi_nums, category_nums , user_nums = split_dataset(dataset_path)


    #--------------get pytorch-style dataset-----------#
    train_dataset, test_dataset, train_sampler, valid_sampler = get_dataset(user_traj_train, user_traj_test, train_nums)
    devices = try_all_gpus()
        

    #----------------Repeat the experiment-------------#
    for idx, seed in enumerate(random.sample(range(0, 1000), args.times)):
        
        #---------------Repeatability settings---------------#
        seed = 666
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
        
        #----------Building networks and optimizers----------#
        
        student_timeaware_embedding = LstmTimeAwareEmbedding(args.embed_size, poi_nums, category_nums)
        student_encoder = StudentEncoder(student_timeaware_embedding, args.embed_size*2, user_nums)

        if args.encoding_type == 'position':
            teacher_encoding_layer = PositionalEncoding(args.embed_size, max_seq_len=800)
        elif args.encoding_type == 'temporal':
            teacher_encoding_layer = TemporalEncoding(args.embed_size)
        else:
            raise Exception('Time encoding is not legal!')
        
        teacher_timeaware_embedding = TransformerTimeAwareEmbedding(teacher_encoding_layer, args.embed_size, poi_nums, category_nums)
        teacher_encoder = TeacherEncoder(teacher_timeaware_embedding, args.embed_size*2, args.num_layers, args.num_heads, user_nums)

        model = TulNet(student_encoder, teacher_encoder)
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        #-------------Start training and logging-------------#
        logger.info('The {} round, start training with random seed {}'.format(idx, seed))
        current_time = time.time()

        avg_train_losses, avg_valid_losses = train_model(train_dataset, train_sampler, valid_sampler, model, optimizer, user_traj_train, devices, args, logger)
        #loss_with_plot(avg_train_losses, avg_valid_losses, args.dataset)

        model.load_state_dict(torch.load('./project/temp/'+ args.dataset +'_checkpoint.pt'))
        test_model(test_dataset, model, devices, args, logger)

        logger.info("Total time elapsed: {:.4f}s".format(time.time() - current_time))
        logger.info('Fininsh trainning in seed {}\n'.format(seed))


if __name__ == '__main__':
    main()