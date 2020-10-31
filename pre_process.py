import pickle
from collections import Counter
import json
import jieba
import nltk
from tqdm import tqdm
import numpy as np

from config import train_filename,valid_filename,maxlen_in,\
    vocab_file,  maxlen_out, data_file, sos_id, eos_id, n_src_vocab, \
     unk_id
from utils import normalizeString, encode_text


def build_vocab(token, word2idx, idx2char):
    if token not in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def getCQpair(filename):
    with open(filename, 'r') as fh:
        train_txt = json.load(fh)
    questions = []
    contexts = []
    for i in range(len(train_txt)):
        questions.append(train_txt[i]["question"])
        contexts_per = ""
        for j in range(len(train_txt[i]["context"])):
            contexts_per += train_txt[i]["context"][j][1][0]

        contexts.append(contexts_per)

    # train_data = []
    # for i in range(len(contexts)):
    #     train_data.append(["ask_question", contexts[i], questions[i]])
    #
    # train_df = pd.DataFrame(train_data)
    # train_df.columns = ["prefix", "input_text", "target_text"]

    return contexts,questions

def process(file):
    print('processing {}...'.format(file))

    contexts, questions = getCQpair(file)
    print('contexts length',len(contexts))
    print('questions length',len(questions))
    word_freq = Counter()
    lengths = []

    for line in tqdm(contexts):
        sentence = line.strip()

        # sentence_en = sentence.lower()
        tokens = [s for s in nltk.word_tokenize(sentence)]
        word_freq.update(list(tokens))
        vocab_size = n_src_vocab

        lengths.append(len(tokens))

    words = word_freq.most_common(vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<sos>'] = 1
    word_map['<eos>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:100])
    #
    # n, bins, patches = plt.hist(lengths, 50, density=True, facecolor='g', alpha=0.75)
    #
    # plt.xlabel('Lengths')
    # plt.ylabel('Probability')
    # plt.title('Histogram of Lengths')
    # plt.grid(True)
    # plt.show()

    word2idx = word_map
    idx2char = {v: k for k, v in word2idx.items()}

    return word2idx, idx2char,word_freq


def get_data(in_file):
    contexts, questions = getCQpair(in_file)

    samples = []

    for i in tqdm(range(len(contexts))):

        tokens = [s.strip() for s in nltk.word_tokenize(contexts[i])]
        in_data = encode_text(word2idx_dict, tokens)


        q_tokens = [s.strip() for s in nltk.word_tokenize(questions[i])]
        out_data = [sos_id] + encode_text(word2idx_dict, q_tokens) + [eos_id]

        if  len(in_data) < maxlen_in and len(out_data) < maxlen_out:
            samples.append({'in': in_data, 'out': out_data})

    return samples

def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # filtered_elements = [k for (k, v) in counter if v > limit]

    filtered_elements = [k for k,v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))

    token2idx_dict = {token: idx+2 for idx, token in enumerate(
        embedding_dict.keys(), 2)}

    token2idx_dict['<pad>'] = 0
    token2idx_dict['<sos>'] = 1
    token2idx_dict['<eos>'] = 2
    token2idx_dict['<unk>'] = 3
    embedding_dict['<pad>'] = [0. for _ in range(vec_size)]
    embedding_dict['<sos>'] = [0. for _ in range(vec_size)]
    embedding_dict['<eos>'] = [0. for _ in range(vec_size)]
    embedding_dict['<unk>'] = [0. for _ in range(vec_size)]

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}

    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    idx2token_dict = {idx: token for token, idx in token2idx_dict.items()}

    return emb_mat, token2idx_dict, idx2token_dict

if __name__ == '__main__':
    word2idx, idx2word,word_counter = process(train_filename)


    # vocab = pickle.load(open( "vocab.pkl", "rb" ))
    # print(vocab)
    # word2idx = vocab['dict']['word2idx']
    # idx2word = vocab['dict']['idx2word']
    word_emb_mat, word2idx_dict, idx2word_dict = get_embedding(word_counter, "word", emb_file= "glove.840B.300d.txt",
                                                               size=int(2.2e6), vec_size=300,
                                                               token2idx_dict=word2idx)
    print(len(word_emb_mat))
    print(len(word2idx_dict))
    print(len(idx2word_dict))

    data = {
        'dict': {
            'word2idx': word2idx_dict,
            'idx2word': idx2word_dict,

        }
    }
    with open('word_emb.p', 'wb') as file:
        pickle.dump(word_emb_mat, file)

    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)

    train = get_data(train_filename)
    valid = get_data(valid_filename)

    data = {
        'train': train,
        'valid': valid
    }

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open(data_file, 'wb') as file:
        pickle.dump(data, file)
