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
##### (1)  download chinese bert model

* git clone https://huggingface.co/bert-base-chinese

##### (2)  install transformer

* pip install transformers
* conda install -c huggingface transformers

##### (3)  change your own data file path

* form file config/config.py change bert_data_path, bert_tag_path, bert_model_path, bert_vocab_path

##### (4) learning rate option

* [2e-5, 3e-5, 4e-5, 5e-5]

result
----

|datasets |  P  |  R  |  F1  |
|-------- |-----|-----|-----|
|  weibo  |0.6866|0.6932|0.6899|
| Resume  |0.9214|0.9497|0.9353|
|MSRA|0.9616|0.9104|0.9353|
|ontonote4|0.8344|0.7404|0.7846|

reference
----
* https://github.com/huggingface/transformers
* https://arxiv.org/abs/1810.04805
* https://ieeexplore.ieee.org/document/818041
