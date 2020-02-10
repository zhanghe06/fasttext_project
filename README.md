# fasttext

https://fasttext.cc

对比 Word2Vec

word2vec是Google与2013年开源推出的一个用于获取word vecter的工具包，利用神经网络为单词寻找一个连续向量表示。

fasttext 使用了n-gram模型，可以更好的表达词前后之间的关系

```
virtualenv .env -p python3
source .env/bin/activate
cd downloads
wget --no-check-certificate https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
unzip v0.9.1.zip
cd fastText-0.9.1
make
pip install .
cd ..
wget --no-check-certificate https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
wc -l cooking.stackexchange.txt
head -n 12404 cooking.stackexchange.txt > cooking.train
tail -n 3000 cooking.stackexchange.txt > cooking.valid
```

## Text classification（文本分类）

```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train")
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   69260 lr:  0.000000 loss: 10.049636 ETA:   0h 0m
>>> model.save_model("model_cooking.bin")
```

precision and recall（精度和召回率）
```
>>> model.predict("Why not put knives in the dishwasher?")
(('__label__food-safety',), array([0.06169952]))
>>> model.predict("Why not put knives in the dishwasher?", k=5)
(('__label__food-safety', '__label__baking', '__label__bread', '__label__equipment', '__label__substitutions'), array([0.06169952, 0.06087842, 0.03843379, 0.03824816, 0.03086694]))

➜  downloads git:(master) ✗ grep "Why not put knives in the dishwasher?" cooking.train
__label__equipment __label__cleaning __label__knives Why not put knives in the dishwasher?
```
模型预测的五分之一标签是正确的，精确度为0.20。
在三个真实标签中，模型仅预测了一个，召回率为0.33。

```
>>> model.test("cooking.valid")
(3000, 0.14833333333333334, 0.06414876747873721)
```

模型优化

一、数据优化

混合大小写字母全部统一为小写
```
cat cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > cooking.preprocessed.txt
head -n 12404 cooking.preprocessed.txt > cooking.train
tail -n 3000 cooking.preprocessed.txt > cooking.valid
```

二、调整参数

1、epoch

epoch 范围: \[5 - 50\]

```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", epoch=25)
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   61773 lr:  0.000000 loss:  7.660386 ETA:   0h 0m
>>> model.test("cooking.valid")
(3000, 0.4533333333333333, 0.1960501657777137)
```

可以看到精度和召回率都提高了

2、learning rate

lr 范围: \[0.1 - 1.0\]

```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0)
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   69435 lr:  0.000000 loss:  6.735564 ETA:   0h 0m
>>> model.test("cooking.valid")
(3000, 0.5173333333333333, 0.22372783624044976)
```

```
>>> import fasttext
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25)
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   62800 lr:  0.000000 loss:  4.066584 ETA:   0h 0m
>>> model.test("cooking.valid")
(3000, 0.5593333333333333, 0.2418913074816203)
```

3、word n-grams（字母组）

word n-grams 范围: \[1 - 5\]

```
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   57989 lr:  0.000000 loss:  3.289264 ETA:   0h 0m
>>> model.test("cooking.valid")
(3000, 0.5663333333333334, 0.24491855268848206)
```


Scaling things up（扩容）

损耗函数: hierarchical softmax
```
>>> model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs')
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread: 1705276 lr:  0.000000 loss:  2.263285 ETA:   0h 0m
>>> model.test("cooking.valid")
(3000, 0.5416666666666666, 0.23425111719763586)
```

Multi-label classification（多标签分类）
```
>>> model = fasttext.train_supervised(input="cooking.train", lr=0.5, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='ova')
Read 0M words
Number of words:  14543
Number of labels: 735
Progress: 100.0% words/sec/thread:   90007 lr:  0.000000 loss:  4.572579 ETA:   0h 0m
>>> model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
(('__label__baking', '__label__bananas', '__label__bread'), array([1.00001001, 0.9450047 , 0.9450047 ]))
```

## Word representations（单词特征）

```
mkdir data
wget -c http://mattmahoney.net/dc/enwik9.zip -P data
unzip data/enwik9.zip -d data
perl wikifil.pl data/enwik9 > data/fil9
```

```
>>> import fasttext
>>> model = fasttext.train_unsupervised('data/fil9')
>>> model.words
>>> model.get_word_vector("the")
>>> model.save_model("result/fil9.bin")
>>> model = fasttext.load_model("result/fil9.bin")
```

维度（dim）控制向量的大小，向量越大，它们可以捕获的信息越多，但需要学习的数据也更多。
但是，如果它们太大，则训练起来会越来越困难。
默认情况下，我们使用100个尺寸，但是100-300范围内的任何值都非常受欢迎。

子词（subwords）是一个单词中包含的所有子串，介于最小大小（minn）和最大大小（maxn）之间。
默认情况下，我们使用3到6个字符之间的所有子词，但其他范围可能更适合于不同的语言

```
>>> import fasttext
>>> model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)
```

学习率（lr）。学习率越高，模型收敛到解决方案的速度就越快，但存在过度拟合数据集的风险。
默认值为0.05，这是一个很好的折衷。如果您想使用它，我们建议保持在\[0.01，1\]范围内：
```
>>> import fasttext
>>> model = fasttext.train_unsupervised('data/fil9', epoch=1, lr=0.5)
```

fastText是多线程的，默认情况下使用12个线程。如果CPU内核较少（例如4个），则可以使用线程标志轻松设置线程数
```
>>> import fasttext
>>> model = fasttext.train_unsupervised('data/fil9', thread=4)
```

相似度、类比
```
>>> model.get_word_vector("enviroment")
>>> model.get_nearest_neighbors('asparagus')
>>> model.get_nearest_neighbors('pidgey')
>>> model.get_nearest_neighbors('enviroment')
>>> model.get_analogies("berlin", "germany", "france")
>>> model.get_analogies("psx", "sony", "nintendo")
>>> model.get_nearest_neighbors('gearshift')
```
