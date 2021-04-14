from config.metric import get_ner_fmeasure as measure
from config.config import params as hp
from model.Bert_lstm import Bertencoder
from pro_data.get_data import load_data
from torch.nn import DataParallel
from config.AdamW import AdamW
from torch.optim import Adam, SGD
import torch


def train():

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    # load train data
    train_data = load_data(hp.batch_size, use_gpu, 'train')
    train_x, train_y, train_seg_ids, train_mask = train_data.get_batch_data()

    # load test data
    test_data = load_data(hp.batch_size, use_gpu, 'test')
    label_y = test_data.label_y
    test_x, test_y, test_seg_ids, test_mask = test_data.get_batch_data()

    # model
    model = Bertencoder(hp.bert_dim, hp.output_dim, hp.num_layers, hp.rnn_dim)
    if use_gpu:
        # use more gpu train model
        #model = Dataparallel(model, device_ids=[0, 1, 2, 3])
        model.to(device)

    # define optimizer
    #optimizer = AdamW(model.parameters(), lr=hp.adamw_lr, weight_decay=0.001)
    optimizer = Adam(model.parameters(), lr=hp.adam_lr)

    # train model
    for i in range(hp.epoch):
        epoch_loss = 0
        for j in range(len(train_x)):
            if use_gpu:
                train_x[j], train_y[j] = train_x[j].to(device), train_y[j].to(device)
                train_seg_ids[j], train_mask[j] = train_seg_ids[j].to(device), train_mask[j].to(device)
            #loss = model(train_x[j], train_y[j], train_seg_ids[j], train_mask[j]).mean()                # dataparallel
            loss = model(train_x[j], train_y[j], train_seg_ids[j], train_mask[j])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("epoch:\t{}\t\titeration:\t{}\t\tbatch_loss:\t{}".format(i + 1, j + 1, round(loss.item(), 4)))
        print("epoch_loss:\t{}".format(round(epoch_loss/(j+1), 4)))

        # test model
        dev_loss = 0
        predict_idx_label = []

        for h in range(len(test_x)):
            if use_gpu:
                test_x[h], test_y[h] = test_x[h].to(device), test_y[h].to(device)
                test_seg_ids[h], test_mask[h] = test_seg_ids[h].to(device), test_mask[h].to(device)
            with torch.no_grad():
                #loss, pre_y = model.module.test(test_x[h], test_y[h], test_seg_ids[h], test_mask[h])     # dataparallel
                loss, pre_y = model.test(test_x[h], test_y[h], test_seg_ids[h], test_mask[h])
            #dev_loss += loss.mean().item()        # dataparallel
            dev_loss += loss.item()

            predict_idx_label.append(pre_y)
            #print('test_iteration:\t{}\t\tloss:\t{}'.format(h+1, round(loss.mean().item(), 4)))          # dataparallel
            print('test_iteration:\t{}\t\tloss:\t{}'.format(h+1, round(loss.item(), 4)))

        predict_label = test_data.trans2tag(predict_idx_label)

        a, p, r, f = measure(label_y, predict_label)
        print('train_loss:\t{}\t\tdev_loss:\t{}'.format(round(epoch_loss/(j+1), 4), round(dev_loss/(h+1), 4)))
        print('A:\t{}\t\tP:\t{}\t\tR:\t{}\t\tF:\t{}'.format(a, p, r, f))
        print('*******************************************************')


if __name__ == '__main__':
    train()











