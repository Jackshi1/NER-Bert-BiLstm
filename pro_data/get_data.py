from config.config import bert_data_path, bert_tag_path, bert_vocab_path
from pro_data.get_file import load_json_data
import torch
import math


class load_data(object):
    """
    inputs include: character index, token_type_ids, attention_mask
    """
    def __init__(self, batch_size, use_gpu, dtype):
        self.dtype = dtype
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.text_x = []
        self.label_y = []
        self.tag2id = None
        self.vocab = self.load_vocab()
        if dtype == 'train':
            self.get_data()
        else:
            self.get_data()

    def load_vocab(self):
        vocab = dict()
        index = 0
        with open(bert_vocab_path, "r") as f:
            #print(f.read())
            for line in f.readlines():
                token = line.strip()
                vocab[token] = index
                index += 1
        return vocab

    def trans2tag(self, pre_y):
        """
        input 3D to output 2D
        index to tag
        """
        id2tag = {self.tag2id[tag]:tag for tag in self.tag2id}
        predict_label = []
        for each_batch in pre_y:
            for each_sent in each_batch:
                predict_label.append([id2tag.get(idx, 0) for idx in each_sent])
        return predict_label

    def get_data(self):
        data = load_json_data(bert_data_path)
        self.tag2id = load_json_data(bert_tag_path)

        if self.dtype == 'train':
            self.text_x = data['train_x']
            self.label_y = data['train_y']
        else:
            self.text_x = data['test_x']
            self.label_y = data['test_y']

    def get_batch_data(self):
        self.get_data()
        mask = []
        segment_ids = []
        text_idx = []
        label_idx = []

        len_data = len(self.text_x)
        train_iter = math.ceil(len_data/self.batch_size)
        print(self.dtype + "total epoch = ", train_iter)
        for i in range(train_iter):
            start, end = i*self.batch_size, (i+1)*self.batch_size
            if end < len_data:
                text_batch = self.text_x[start:end]
                label_batch = self.label_y[start:end]
            else:
                text_batch = self.text_x[start:]
                label_batch = self.label_y[start:]

            pad_text_batch, pad_label_batch, pad_seg_ids, pad_mask = self.padding_batch_data(text_batch, label_batch)

            text_idx.append(pad_text_batch)
            label_idx.append(pad_label_batch)
            segment_ids.append(pad_seg_ids)
            mask.append(pad_mask)
        return text_idx, label_idx, segment_ids, mask

    def padding_batch_data(self, text_batch, label_batch):
        """
        transfer each batch data to idx, and add padding
        get mask
        get segment_ids
        """
        max_batch_len = max([len(sent) for sent in text_batch])+2
        if max_batch_len > 512:
            max_batch_len = 512

        batch_mask = []
        batch_seg_ids = []
        batch_x = []
        batch_y = []

        for text_sent, label_sent in zip(text_batch, label_batch):
            if max_batch_len == 512:
                if len(text_sent) > 510:
                    text_sent = text_sent[:510]
                    label_sent = label_sent[:510]
            pad_len = max_batch_len - len(text_sent) - 2
            batch_seg_ids.append([0]*max_batch_len)
            sent_mask = [1]*(len(text_sent)+2)+[0]*pad_len
            batch_mask.append(sent_mask)

            x = ['[CLS]'] + text_sent + ['[SEP]']
            y = ['[CLS]'] + label_sent + ['[SEP]']

            x_idx = [self.vocab.get(word, 1) for word in x]
            y_idx = [self.tag2id.get(tag, 0) for tag in y]

            batch_x.append(x_idx+[0]*pad_len)
            batch_y.append(y_idx+[0]*pad_len)

        if self.use_gpu:
            batch_x, batch_y, batch_seg_ids, batch_mask = torch.tensor(batch_x).long(), torch.tensor(batch_y).long(),\
                                                          torch.tensor(batch_seg_ids).long(), torch.tensor(batch_mask).long()
        return batch_x, batch_y, batch_seg_ids, batch_mask

