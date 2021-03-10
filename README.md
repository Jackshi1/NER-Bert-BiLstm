Chinese Namede Entity Recognition
==
model: Bert+BiLstm  
loss function: Cross Entropy Error Function<br>

Chinese NER datasets: 
----
*  chinese weibo dataset<br>
*  chinese ontonote4<br>
*  chinese Resume<br>
*  chinese MSRA<br>

how to train
----
##### (1) download chinese bert model

* git lfs install
* git clone https://huggingface.co/bert-base-chinese

##### (2) install transformer

* pip install transformers

##### (3) change your own data file path

* form file config/config.py change bert_data_path, bert_tag_path, bert_model_path, bert_vocab_path

result
----

|datasets |  P  |  R  |  F  |
|-------- |-----|-----|-----|
|  weibo  |     |     |     |
|  MSRA   |     |     |     |
|ontonote4|     |     |     |
| Resume  |     |     |     |

reference
----
* https://github.com/huggingface/transformers
* https://arxiv.org/abs/1810.04805
* https://ieeexplore.ieee.org/document/818041
