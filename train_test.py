import os.path
import time
from loggers import *
from copy import deepcopy
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from metrics import *

def mtlTrain(gru4rec_model, mmoe_model, train_loader, val_loader, test_loader, args):
    device = args.device
    epochs = args.epochs
    early_stop = 5
    path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    loss_function = nn.BCEWithLogitsLoss()
    gru4rec_optimizer = torch.optim.adam(gru4rec_model.parameters(), lr=args.lr)
    mmoe_optimizer = torch.optim.adam(mmoe_model.parameters(), lr=args.lr)
    gru4rec_model.to(device)
    mmoe_model.to(device)
    
    # Early stopping
    patience, eval_loss = 0, 0

    for epoch in range(epochs):
        gru4rec_model.train()
        mmoe_model.train()

        y_train_click_true = []
        y_train_click_predict = []
        y_train_like_true = []
        y_train_like_predict = []

        total_loss, count = 0, 0
        for idx, (x, y1, y2) in enumerate(train_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

            # Get session embeddings from GRU4Rec
            session_embedding = gru4rec_model(x)
            
            # Pass session embeddings to MMOE
            predict = mmoe_model(session_embedding)

            y_train_click_true += list(y1.squeeze().cpu().numpy())
            y_train_like_true += list(y2.squeeze().cpu().numpy())
            y_train_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2

            gru4rec_optimizer.zero_grad()
            mmoe_optimizer.zero_grad()
            loss.backward()
            gru4rec_optimizer.step()
            mmoe_optimizer.step()

            total_loss += float(loss)
            count += 1

        click_auc = roc_auc_score(y_train_click_true, y_train_click_predict)
        like_auc = roc_auc_score(y_train_like_true, y_train_like_predict)
        print("Epoch %d train loss is %.3f, click auc is %.3f and like auc is %.3f" % (epoch + 1, total_loss / count, click_auc, like_auc))

        # Validation
        gru4rec_model.eval()
        mmoe_model.eval()

        y_val_click_true = []
        y_val_like_true = []
        y_val_click_predict = []
        y_val_like_predict = []

        total_eval_loss, count_eval = 0, 0
        for idx, (x, y1, y2) in enumerate(val_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

            # Get session embeddings from GRU4Rec
            session_embedding = gru4rec_model(x)
            
            # Pass session embeddings to MMOE
            predict = mmoe_model(session_embedding)

            y_val_click_true += list(y1.squeeze().cpu().numpy())
            y_val_like_true += list(y2.squeeze().cpu().numpy())
            y_val_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2

            total_eval_loss += float(loss)
            count_eval += 1

        click_auc_val = roc_auc_score(y_val_click_true, y_val_click_predict)
        like_auc_val = roc_auc_score(y_val_like_true, y_val_like_predict)
        print("Epoch %d validation loss is %.3f, click auc is %.3f and like auc is %.3f" % (epoch + 1, total_eval_loss / count_eval, click_auc_val, like_auc_val))

        if total_eval_loss < eval_loss or epoch == 0:
            eval_loss = total_eval_loss
            patience = 0
            torch.save(gru4rec_model.state_dict(), path.replace("best_model", "gru4rec_best_model"))
            torch.save(mmoe_model.state_dict(), path)
        else:
            patience += 1

        if patience >= early_stop:
            break

    # Load best models
    gru4rec_model.load_state_dict(torch.load(path.replace("best_model", "gru4rec_best_model")))
    mmoe_model.load_state_dict(torch.load(path))

    # Testing
    gru4rec_model.eval()
    mmoe_model.eval()

    y_test_click_true = []
    y_test_like_true = []
    y_test_click_predict = []
    y_test_like_predict = []

    for idx, (x, y1, y2) in enumerate(test_loader):
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)

        # Get session embeddings from GRU4Rec
        session_embedding = gru4rec_model(x)
        
        # Pass session embeddings to MMOE
        predict = mmoe_model(session_embedding)

        y_test_click_true += list(y1.squeeze().cpu().numpy())
        y_test_like_true += list(y2.squeeze().cpu().numpy())
        y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
        y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())

    click_auc_test = roc_auc_score(y_test_click_true, y_test_click_predict)
    like_auc_test = roc_auc_score(y_test_like_true, y_test_like_predict)
    print("Test click auc is %.3f and like auc is %.3f" % (click_auc_test, like_auc_test))



#-------------------------SEQUENCE------------------------#
def SeqTrain(epochs, model, train_loader, val_loader, writer, args):
    if args.is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(args.device)
    if args.is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model,  find_unused_parameters=True,device_ids=[args.local_rank], output_device=args.local_rank)
    best_metric = 0
    all_time = 0
    val_all_time = 0
    for epoch in range(epochs):
        since = time.time()
        optimizer = SequenceTrainer(epoch, model, train_loader, optimizer, writer, args)
        tmp = time.time() - since
        print('one epoch train:', tmp)
        all_time += tmp
        val_since = time.time()
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer, args)
        val_tmp = time.time() - val_since
        print('one epoch val:', val_tmp)
        val_all_time += val_tmp
        if args.is_pretrain == 0 and 'acc' in args.task_name:
            if metrics['NDCG@20'] >= 0.0193:
                break
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            state_dict = model.state_dict()
            if 'life' in args.task_name:
                torch.save(state_dict, os.path.join(args.save_path,
                                                         '{}_{}_seed{}_task_{}_best_model.pth'.format('sequence',
                                                                                                      args.model_name,
                                                                                                      args.seed,
                                                                                                      args.task)))
            else:
                torch.save(state_dict, os.path.join(args.save_path, '{}_{}_seed{}_is_pretrain_{}_best_model_lr{}_wd{}_block{}_hd{}_emb{}.pth'.format(args.task_name, args.model_name, args.seed, args.is_pretrain,
                                                                                                                              args.lr, args.weight_decay, args.block_num, args.hidden_size, args.embedding_size)))
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model


def SequenceTrainer(epoch, model, dataloader, optimizer, writer, args): #schedular,
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for data in dataloader:
        optimizer.zero_grad()
        data = [x.to(args.device) for x in data]
        seqs, labels = data
        logits = model(seqs) # B x T x V
        if 'cold' in args.task_name or ('life_long' in args.task_name and args.task != 0):
            logits = logits.mean(1)
            labels = labels.view(-1)
        else:
            logits = logits.view(-1, logits.size(-1)) # (B*T) x V
            labels = labels.view(-1)  # B*T

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
    #writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    return optimizer

def Sequence_full_Validate(epoch, model, dataloader, writer, args, test=False):
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0
    with torch.no_grad():
        tqdm_dataloader = dataloader
        for data in tqdm_dataloader:
            data = [x.to(args.device) for x in data]
            seqs, labels = data
            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)
            scores = scores.mean(1)
            metrics = recalls_and_ndcgs_for_ks(scores, labels, args.metric_ks, args)
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
    print(avg_metrics)
    for k in sorted(args.metric_ks, reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k), avg_metrics['NDCG@%d' % k], epoch)
    return avg_metrics

def recalls_and_ndcgs_for_ks(scores, labels, ks, args):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float()).to(args.device)
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count]).to(args.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg

    return metrics