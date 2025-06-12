from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import json
import numpy as np
import argparse

class Scorer():
    def __init__(self,ref,gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.4f"%(m, sc))
                    pass
                total_scores["Bleu"] = np.mean(score)
            else:
                print("%s: %0.4f"%(method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key,value in total_scores.items():
            print('{}:{}'.format(key,value))

def sentence_cleaned(text):
    cleaned_text = text.replace("</s>", "").replace("<s>", "").replace("</n>", "").strip()
    return cleaned_text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default= "./outputs/DDVQA/DDVQA_exp_c40.jsonl")
    parser.add_argument('--gt_path', type=str, default="./utils/DDVQA_eval/c40/test.jsonl")
    args = parser.parse_args()

    gt_dict, ref_dict = {}, {}

    with open(args.predict_path, 'r') as f:
        for line in f:
            ref_data = json.loads(line)
            answer = sentence_cleaned(ref_data['text'])
            ref_dict[ref_data['key']] = [answer]

    with open(args.gt_path, 'r') as f:
        for line in f:
            gt_data = json.loads(line)
            key = list(gt_data.keys())[0]
            gt_dict[key] = gt_data[key]['answer']

    scorer = Scorer(ref_dict, gt_dict)
    scorer.compute_scores()