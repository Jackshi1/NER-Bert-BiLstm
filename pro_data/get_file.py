
import torch
import json


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


def read_label_data(filename, split='\t'):
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


def load_test_data():
    #CN_test = load_json_data('../data/char_data.json')
    char_test = load_json_data('./data/char_test.json')
    return char_test


def load_embedding():
    data = load_json_data('../data/embedding_1.json')
    value = data['weight']
    embedding = torch.tensor(value)
    print(embedding.shape)


def count_sent_len():
    data = load_json_data('../news_data/data/sent_data.json')
    max_len = 0
    count = 0
    for each in data['tag']:
        if len(each) > max_len:
            max_len = len(each)
        count += len(each)
    print('sent max len = ', max_len, '\nsent avg len = ', count/len(data['tag']))


def count_each_label(batch_num):
    data = load_json_data('../news_data/data/bert_data.json')
    batch = 64
    train_tag = data['train_y']
    label_count = {}
    for i in range(batch * batch_num):
        # print(train_tag[i])
        for each_tag in train_tag[i]:
            if each_tag != 'O':
                first_tag, second_tag = each_tag.split('-')
                if second_tag in label_count and first_tag in ['B', 'S']:
                    label_count[second_tag] += 1
                elif second_tag not in label_count:
                    label_count[second_tag] = 1
    return label_count

def func():
    data = load_json_data('../data/char_data.json')
    test = data['test_y']
    dev = data['dev_y']
    print(len(test), len(dev))

if __name__ == '__main__':
    #text, _ = read_label_data('../data/test_ner.data')
    #save_json_data(text, '../data/CN_test.json')
    #load_embedding()
    #count_sent_len()
    #print(count_each_label(1))
    func()

