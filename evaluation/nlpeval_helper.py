import string
import re
import numpy as np
from collections import Counter

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from nlgeval.pycocoevalcap.rouge.rouge import Rouge



def _strip(s):
    return s.strip()

def compute_metrics(hyp_list, ref_list):
    '''
    Based on the original implementation of nlpeval
    '''
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                # print("%s: %0.6f" % (m, sc))
                ret_scores[m] = sc
        else:
            # print("%s: %0.6f" % (method, score))
            ret_scores[method] = score
        if isinstance(scorer, Meteor):
            scorer.close()
    del scorers

    return ret_scores




def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def compute_rolling_rouge_L(summary, references):
    '''
    summary:        [str]  Target passage to be evaluated
    references:     [list] A list of reference strings
    '''
    
    summary_toks = summary.split()
    best_score   = -1
    best_subseq  = None
    em           = False

    for ref in references:
        ref_toks = ref.split()
        if len(summary_toks) < len(ref_toks):
            hypo_dict   = {0: [' '.join(summary_toks)]}
            ref_dict    = {0: [ref]}
        else:
            summary_subseqs = zip(*(summary_toks[i : ] for i in range(len(ref_toks))))
            hypo_dict   = {idx: [' '.join(subseq)] for (idx, subseq) in enumerate(summary_subseqs)}
            ref_dict    = {idx: [ref] for idx in range(len(hypo_dict))}
        mean_score, scores  = Rouge().compute_score(ref_dict, hypo_dict)
        max_idx     = np.argmax(scores)
        max_score   = scores[max_idx]
        if max_score > best_score:
            best_score  = max_score
            best_subseq = hypo_dict[max_idx][0]
        em          = em or ref in summary
    return {
        'rouge-l': best_score, 
        'rouge-l_subseq': best_subseq,
        'em':   int(em)
    }
