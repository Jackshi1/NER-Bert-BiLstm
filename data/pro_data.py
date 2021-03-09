
import json
import random

PAD = '<pad>'
UNK = '<unk>'
PAD2ID = 0
UNK2ID = 1


def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            data.append(line.strip())
    return data


def save_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False)


def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def read_label_data(filename, split=' '):
    """
    加载数据集
    :param filename:
    :return:
    """
    text = []
    tags = []
    lines = load_data(filename)
    sentence = []
    tag = []
    for line in lines:
        print(line)
        token = line.split(split)
        if len(token) == 2:
            sentence.append(token[0])
            tag.append(token[1])
        elif len(token) == 1 and len(sentence) > 0:
            text.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    if len(sentence) > 0:
        text.append(sentence)
        tags.append(tag)
    return text, tags


def read_row_label_data(filename):

    text = []
    tags = []
    lines = load_data(filename)
    for line in lines:
        line_text, line_tag = line.split('\t')
        """
        line_list = line.split(' ')
        if '\t' not in line_list:
            index = int(len(line_list)/2)
            line_text = line_list[:index]
            line_tag = line_list[index:]
        else:
            index = line_list.index('\t')
            line_text = line_list[:index]
            line_tag = line_list[index+1:]
        """
        text.append(line_text.split(' '))
        tags.append(line_tag.split(' '))
    for i in range(len(text)):
        if len(text[i]) != len(tags[i]):
            print('the length not equ!')
    if len(text) != len(lines):
        print('wrong!')
    return text, tags


def pro_dataset():

    tag_dic = {}
    count = 0

    train_x, train_y = read_label_data('./police/train_all_bmes')
    print(train_x)
    #dev_x, dev_y = read_label_data('./msra/dev.ner')
    #test_x, test_y = read_label_data('./msra/test.ner')
    """
    train_x, train_y = read_row_label_data('./resume/train.char.bmes')
    dev_x, dev_y = read_row_label_data('./resume/dev.char.bmes')
    test_x, test_y = read_row_label_data('./resume/test.char.bmes')
    """

    for each_sent in train_y:
        for each_word_tag in each_sent:
            if each_word_tag not in tag_dic:
                tag_dic[each_word_tag] = count
                count += 1
    tag_dic["[SEP]"] = count
    count += 1
    tag_dic["[CLS]"] = count

    """
    data = {
        'train_x': train_x,
        'train_y': train_y,
        'dev_x': dev_x,
        'dev_y': dev_y,
        'test_x': test_x,
        'test_y': test_y
    }
    """
    data = {
        'train_x': train_x,
        'train_y': train_y
    }

    save_json_data(data, './police/bert_data.json')
    save_json_data(tag_dic, './police/bert_tag.json')


def func():
    data = load_json_data('./msra/bert_data.json')
    text = data['test_x']
    tags = data['test_y']

    for i in range(len(text)):
        s = ''
        if len(text[i]) > 400:
            for each_c in text[i]:
                s += each_c
            print(text[i])
            print(tags[i])
            print(s)
    """
        if len(text[i]) != len(tags[i]):
            print(i)
    
        if i == 1883:
            j = 1882*4
            for j in range(1882*4, 1885*4):
                print(text[j])
                print(tags[j])
                print(len(tags[j]))

    for i in range(len(text)):
        if len(text[i]) > 400:
            print(len(text[i]))
            print(text[i])
    """


def pro_msra_data():
    data = load_json_data('./msra/bert_data.json')
    dict_name = ['train_x', 'train_y', 'test_x', 'test_y', 'dev_x', 'dev_y']
    data_new = {}

    for j in range(0, 6, 2):
        text = data[dict_name[j]]
        tag = data[dict_name[j+1]]
        print(dict_name[j], dict_name[j+1])
        print(len(tag), len(text))
        text_x = []
        tag_y = []
        for i in range(len(text)):
            if '℃' in text[i]:
                #print(text[i])
                split_x = []
                split_y = []
                for h in range(len(text[i])-1):
                    split_x.append(text[i][h])
                    split_y.append(tag[i][h])
                    if text[i][h] == '℃' and text[i][h+1] != '／' and h != (len(text[i])-2) and text[i][h+1] != '—':
                        text_x.append(split_x)
                        tag_y.append(split_y)
                        #print(split_y)
                        #print(split_x)
                        split_x = []
                        split_y = []
                    if h == (len(text[i])-2):
                        split_x.append(text[i][h + 1])
                        split_y.append(tag[i][h + 1])
                        #print(split_x)
                        text_x.append(split_x)
                        tag_y.append(split_y)
            elif '；' in text[i]:
                split_x = []
                split_y = []
                for h in range(len(text[i])):
                    split_x.append(text[i][h])
                    split_y.append(tag[i][h])
                    if text[i][h] == '；':
                        text_x.append(split_x)
                        tag_y.append(split_y)
                        split_x = []
                        split_y = []
                    if h == (len(text[i])-1):
                        text_x.append(split_x)
                        tag_y.append(split_y)
            elif len(text[i]) > 400:
                #print(text[i])
                split_x = []
                split_y = []
                for h in range(len(text[i])):
                    split_x.append(text[i][h])
                    split_y.append(tag[i][h])
                    if text[i][h] == '，' or text[i][h] == '）':
                        text_x.append(split_x)
                        tag_y.append(split_y)
                        split_x = []
                        split_y = []
                    if h == (len(text[i])-1):
                        text_x.append(split_x)
                        tag_y.append(split_y)

            else:
                text_x.append(text[i])
                tag_y.append(tag[i])
        data_new[dict_name[j]] = text_x
        data_new[dict_name[j+1]] = tag_y

    save_json_data(data_new, './msra/bert_data.json')

def count_iteration():
    data = load_json_data('./resume/bert_data.json')
    test = data['test_x']
    i = len(test) / 32
    j = len(test) % 32
    print(i, j)


class process_police(object):
    def __init__(self, load_data_path, save_data_path):
        self.train_x = []
        self.train_y = []
        self.tag_dic = {}
        self.load_data_path = load_data_path
        self.save_data_path = save_data_path

    def load_label_data(self):
        """
        加载数据集
        :param load_file_path:
        :return:
        """
        lines = load_data(self.load_data_path)
        sentence = []
        tag = []
        for line in lines:
            token = line.split('\t')
            if len(token) == 2:
                sentence.append(token[0])
                tag.append(token[1])
            elif len(token) == 1 and len(sentence) > 0:
                self.train_x.append(sentence)
                self.train_y.append(tag)
                sentence = []
                tag = []
        if len(sentence) > 0:
            self.train_x.append(sentence)
            self.train_y.append(tag)

    def split_sents(self):
        self.load_label_data()
        text = []
        tags = []
        for i in range(len(self.train_x)):
            sentence = []
            tag = []
            for j in range(len(self.train_x[i])):
                if self.train_x[i][j] != '。':
                    sentence.append(self.train_x[i][j])
                    tag.append(self.train_y[i][j])
                else:
                    sentence.append(self.train_x[i][j])
                    tag.append(self.train_y[i][j])
                    text.append(sentence)
                    tags.append(tag)
                    sentence = []
                    tag = []
        self.train_x = text
        self.train_y = tags

    def pros_label(self):
        self.split_sents()
        for h in range(len(self.train_y)):
            start_index = 0
            end_index = -1
            tag = 'none'
            print(self.train_y[h])
            for i in range(len(self.train_y[h])):
                index_span = end_index-start_index
                if tag == 'none':                                # tag=='none' mean there not tag right now
                    if self.train_y[h][i] != 'O':                # start new tag
                        tag = self.train_y[h][i]
                        start_index = i
                        end_index = i
                else:                                            # tag != 'none' mean there is tag right now
                    if tag == self.train_y[h][i] or i == (len(self.train_y[h])-1):
                        end_index = i                            # continue this label
                        if i == (len(self.train_y[h])-1):        # this label end up with sentence
                            if index_span == 0:
                                self.train_y[h][i] = 'S-' + tag.split('-')[1]
                            elif index_span == 1:
                                self.train_y[h][start_index] = 'B-' + tag.split('-')[1]
                                self.train_y[h][end_index] = 'E-' + tag.split('-')[1]
                            elif index_span > 1:
                                for j in range(start_index + 1, end_index):
                                    self.train_y[h][j] = 'M-' + tag.split('-')[1]
                                self.train_y[h][start_index] = 'B-' + tag.split('-')[1]
                                self.train_y[h][end_index] = 'E-' + tag.split('-')[1]

                    elif tag != self.train_y[h][i]:              # the old tag stop ——new start  ——none tags
                        if index_span == 0:
                            self.train_y[h][i] = 'S-'+tag.split('-')[1]
                        elif index_span == 1:
                            self.train_y[h][start_index] = 'B-' + tag.split('-')[1]
                            self.train_y[h][end_index] = 'E-'+tag.split('-')[1]
                        elif index_span > 1:
                            for j in range(start_index+1, end_index):
                                self.train_y[h][j] = 'M-'+tag.split('-')[1]
                            self.train_y[h][start_index] = 'B-' + tag.split('-')[1]
                            self.train_y[h][end_index] = 'E-'+tag.split('-')[1]
                        if self.train_y[h][i] != 'O':            # old label end and new start a new label
                            start_index = i
                            end_index = i
                            tag = self.train_y[h][i]
                        else:                                    # old label end and there is not a new label
                            start_index = i
                            end_index = i-1
                            tag = 'none'

    def split_long_sent(self, path):
        data = load_json_data()
        train_x = data['train_x']
        train_y = data['train_y']
        new_train_x =[]
        new_train_y =[]
        for i in range(len(train_y)):
            if len(train_y[i]) > 510 and '，' in train_y[i]:
                #print(train_x[i])
                sentence = []
                tags = []
                for j in range(len(train_y[i])):
                    if train_x[i][j] == ',' and train_y[i][j] == 'O':
                        sentence.append(train_x[i][j])
                        tags.append(train_y[i][j])
                        new_train_x.append(sentence)
                        new_train_y.append(tags)
                        sentence = []
                        tags = []
                    else:
                        sentence.append(train_x[i][j])
                        tags.append(train_y[i][j])
            else:
                new_train_x.append(train_x[i])
                new_train_y.append(train_y[i])

        self.train_x = new_train_x
        self.train_y = new_train_y

    def random_split_data(self, file_path):
        data = load_json_data(file_path)
        text = data['train_x']
        label = data['train_y']
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        sent_num = len(text)
        test_index = random.sample([i for i in range(sent_num)], 70)
        for j in range(sent_num):
            if j in test_index:
                test_x.append(text[j])
                test_y.append(label[j])
            else:
                train_x.append(text[j])
                train_y.append(label[j])
        new_data = {
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y
        }
        print('train_length: ', len(train_x), 'test_length: ', len(test_x))
        save_json_data(new_data, self.save_data_path+'bert_data1.json')

    def save_data(self):
        self.pros_label()
        count = 0
        train_data = {
            'train_x': self.train_x,
            'train_y': self.train_y
        }

        for each_sent in self.train_y:
            for each_word_tag in each_sent:
                if each_word_tag not in self.tag_dic:
                    self.tag_dic[each_word_tag] = count
                    count += 1
        self.tag_dic["[SEP]"] = count
        count += 1
        self.tag_dic["[CLS]"] = count

        data_path = save_path+'bert_data.json'
        label_path = save_path+'bert_tag.json'

        save_json_data(train_data, data_path)
        save_json_data(self.tag_dic, label_path)

    def print_data(self):
        self.split_long_sent()
        for i in range(len(self.train_x)):
            print(self.train_x[i])
            print(self.train_y[i])
        print(len(self.train_x), len(self.train_y))

    def count_label_num(self, file_path):
        data = load_json_data(file_path)
        train_y = data['train_y']
        test_y = data['test_y']
        label_train_count = 0
        label_test_count = 0
        for each_sent in train_y:
            no_label = ['O'] * len(each_sent)
            if no_label != each_sent:
                label_train_count += 1
        for each_sent in test_y:
            no_label = ['O'] * len(each_sent)
            if no_label != each_sent:
                label_test_count += 1
        print('train_sent_num: ', len(train_y), 'test_sent_num: ', len(test_y),
              'train_label_num: ', label_train_count, 'test_label_num: ', label_test_count)


def fuc():
    data = load_json_data('./police/bert_data1.json')
    train_x = data['train_x']
    train_y = data['train_y']
    for i in range(len(train_x)):
        print(train_x[i])
        print(train_y[i])


if __name__ == '__main__':
    """
    #pro_dataset()
    #func()
    #pro_msra_data()
    #count_iteration()
    load_path = './police/train_all_bmes'
    save_path = './police/'
    data = process_police(load_path, save_path)
    #data.print_data()
    #data.save_data()
    #data.random_split_data('./police/bert_data.json')
    data.count_label_num('./police/bert_data1.json')
    """
    #fuc()