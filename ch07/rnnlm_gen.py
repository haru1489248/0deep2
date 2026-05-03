import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        """
        start_id: 最初に与える単語ID
        sample_size: サンプリングする単語数
        skip_ids: サンプリングして欲しくない単語IDリスト
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1) # ミニバッチ想定で実装されているpredictに合わせて2次元にする
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), p=p)

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
