import random
class Batch:
    def __init__(self):
        self.inputs = []
        self.inputs_length = []
        self.targets = []
        self.targets_length = []

def get_sentence_int_to_vocab(path1,path2):

    sentence_vocab = []

    f = open(path1, encoding="utf-8")
    g = open(path2,encoding="utf-8")
    for line in f.readlines():
        if line != '\n':
            if len(line.strip().split('\t')) != 0:
                sentence_vocab.append(line.strip().split("\t")[0])
    for lines in g.readlines():
        if lines != '\n':
            if len(lines.strip().split('\t')) != 0:
                sentence_vocab.append(lines.strip().split("\t")[0])
    vocab = list(set(sentence_vocab))
    print("词表大小：",len(vocab))

    tags_vocab = ["O","B-body","I-body","E-body","B-symp","I-symp",
                  "E-symp","B-dise","I-dise","E-dise","B-chec",
                  "I-chec","E-chec","B-cure","I-cure","E-cure"]

    sentence_int_to_vocab = {}
    sentence_vocab_to_int = {}
    tags_int_to_vocab = {}
    symbols1 = {0: "<PAD>", 1: "<UNK>", 2: "<GO>", 3: "<EOS>"}
    symbols2 = {"<PAD>":0, "<UNK>":1 , "<GO>":2 , "<EOS>":3}
    for index_no, word in enumerate(vocab):
        sentence_int_to_vocab[index_no] = word
    sentence_int_to_vocab.update(symbols1)
    for index_nos, words in enumerate(vocab):
        sentence_vocab_to_int[words] = index_nos
    sentence_vocab_to_int.update(symbols2)

    for index_no, word in enumerate(tags_vocab):
        tags_int_to_vocab[index_no] = word
    tags_vocab_to_int = {word: index_no for index_no, word in tags_int_to_vocab.items()}

    return sentence_int_to_vocab,sentence_vocab_to_int,tags_vocab_to_int,tags_int_to_vocab
def get_data(path):
    sentence_int_to_vocab, sentence_vocab_to_int, tags_vocab_to_int, tags_int_to_vocab = get_sentence_int_to_vocab("dataset/train.txt","dataset/test.txt")
    f = open(path,encoding="utf-8")
    f = f.readlines()
    dataset = []
    sent_ = []
    tag_ = []
    for line in f:
        if line != '\n':
            if len(line.strip().split('\t')) != 0:
                fields = line.strip().split('\t')
                char = fields[0]
                label = fields[-1]
                sent_.append(sentence_vocab_to_int[char])
                tag_.append(tags_vocab_to_int[label])
        elif len(sent_) != 0 and len(tag_) != 0:
            sentence = []
            sentence.append(sent_)
            sentence.append(tag_)
            dataset.append(sentence)
            sent_, tag_ = [], []
    print("训练集大小",len(dataset))

    return dataset

def createBatch(samples):

    batch = Batch()
    batch.inputs_length = [len(sample[0]) for sample in samples]
    batch.targets_length = [len(sample[1]) for sample in samples]
    max_source_length = max(batch.inputs_length)
    max_target_length = max(batch.targets_length)

    for j,sample in enumerate(samples):
        source= sample[0]
        batch.inputs.append(source+[0]*(max_source_length - len(source)))
        target = sample[1]
        batch.targets.append(target+[0]*(max_target_length - len(target)))

    return batch

def getBatches(data,batch_size):
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0,data_len,batch_size):
            yield data[i:min(i+batch_size,data_len)]

    for sample in genNextSamples():
        batch = createBatch(sample)
        batches.append(batch)
    return batches

