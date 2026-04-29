import numpy as np

# TODO: 後で詳しく出てくるはず
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

# 前処理用の関数
# テキストに含まれる単語をid変換する
def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 共起行列を作成する関数
# 分布仮説（単語の意味は、周囲の単語によって形成される）に基づいて単語をベクトル表現するためにカウントベースの手法を用いる方法
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# cos類似度の計算
# ゼロ除算を防ぐために非常に小さい数をたす。ゼロでない場合は丸め誤差で吸収される。
def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)

    return np.dot(nx, ny)

# cos類似度をランキング表示する関数
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    query: クエリ（単語）
    word_to_id: 単語から単語IDのディクショナリ
    id_to_word: 単語IDから単語のディクショナリ
    word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    top: 上位何位まで表示するか
    """
    # クエリを取り出す
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果か、その値を高い順に出力
    count = 0
    # argsort()はNumpy配列の要素を小さい順にソートする。（インデックスを返す）
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

# TODO: 確率については後で詳しく出てくるらしい
# PPMI（Positive PMI）を計算する関数
# 単語の出現回数を単語が共起した回数に近似して計算されている
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps) # -infのエラーを避けるため、epsを足している
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1

                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))

    return M

# CBOWのコンテキストとターゲットを作成する関数
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

# コンテキストとターゲットをone-hot表現に変換する関数
def convert_one_hot(corpus, vocab_size):
    '''one-hot表現への変換
    :corpus: 単語IDのリスト（1次元もしくは2次元のNumPy配列）
    :vocab_size: 語彙数
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot
