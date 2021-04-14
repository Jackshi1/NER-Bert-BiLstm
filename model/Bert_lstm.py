from transformers import AutoModel
from config.config import bert_model_path
from torch import nn


class Bertencoder(nn.Module):
    def __init__(self, bert_dim, output_dim, num_layers, rnn_dim):
        super(Bertencoder, self).__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_path)
        self.rnn = nn.LSTM(bert_dim, rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(rnn_dim*2, output_dim)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x, y, seg_ids, mask):
        """
        text:[batch_size, each_sent_len]
        label:[batch_size, max_batch_sent_len]
        """

        bert_output = self.bert_model(x, token_type_ids=seg_ids, attention_mask=mask)
        lstm_output, _ = self.rnn(bert_output.last_hidden_state)
        output = self.linear(lstm_output).transpose(1, 2)
        loss = self.loss_function(output, y)
        return loss

    def test(self, x, y, seg_ids, mask):

        bert_output = self.bert_model(x, token_type_ids=seg_ids, attention_mask=mask)
        lstm_output, _ = self.rnn(bert_output.last_hidden_state)
        output = self.linear(lstm_output).transpose(1, 2)
        loss = self.loss_function(output, y)
        predict = self.predict_label(output.transpose(1, 2), mask)
        return loss, predict

    def predict_label(self, output, mask):
        """
        output: batch_size, max_sentence_len, tag_num
        mask: same label data
        batch_label: batch_size, sentence_label
        """
        pre_label = []
        pre_label_pad = output.argmax(2)
        for i in range(mask.shape[0]):
            sent_label = []
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    sent_label.append(pre_label_pad[i, j].item())
            sent_label = sent_label[1:len(sent_label)-1]
            pre_label.append(sent_label)

        return pre_label











