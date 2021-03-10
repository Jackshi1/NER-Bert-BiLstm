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
* conda install -c huggingface transformers

##### (3) change your own data file path

* form file config/config.py change bert_data_path, bert_tag_path, bert_model_path, bert_vocab_path

result
----

|datasets |  P  |  R  |  F  |
|-------- |-----|-----|-----|
|  weibo  |0.6558|0.6304|0.6429|
|  MSRA   |     |     |     |
|ontonote4|     |     |     |
| Resume  |0.9214|0.9497|0.9353|

reference
----
* https://github.com/huggingface/transformers
* https://arxiv.org/abs/1810.04805
* https://ieeexplore.ieee.org/document/818041
