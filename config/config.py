import os

bert_data_path = os.path.join(os.getcwd(), './data/weibo/bert_data.json')
bert_tag_path = os.path.join(os.getcwd(), './data/weibo/bert_tag.json')
bert_model_path = os.path.join(os.getcwd(), './bert_model')
bert_vocab_path = os.path.join(os.getcwd(), './bert_model/vocab.txt')


class params:
    bert_dim = 768
    output_dim = 30
    batch_size = 32
    adam_lr = 3e-5
    adamw_lr = 5e-5
    epoch = 20
    num_layers = 1
    rnn_dim = 200
