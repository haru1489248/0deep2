import sys
sys.path.append('..')
from common.util import cos_similarity, preprocess, create_co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']] # youの単語ベクトル
c1 = C[word_to_id['i']] # iの単語ベクトル
print(cos_similarity(c0, c1)) # cos類似度 0.7071067758832467

