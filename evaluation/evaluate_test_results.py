import json, os
import argparse

from nlpeval_helper import (
    compute_metrics,
    metric_max_over_ground_truths, 
    exact_match_score, 
    f1_score,
)


parser = argparse.ArgumentParser()
parser.add_argument("--result_file",  type=str)
parser.add_argument("--target_file",  type=str)
args = parser.parse_args()



hypo_file   = args.result_file
source_file = args.target_file
res_file    = 'eval_result.txt'



with open(hypo_file, 'r') as fp:
    predictions = fp.readlines()
    print('# of predictions: ', len(predictions))
with open(source_file, 'r') as fp:
    references  = fp.readlines()
    references1 = references[::2]
    references2 = references[1::2]
assert len(predictions)==len(references1) and len(references1)==len(references2), \
        '# of hypos != # of references / 2'


hyp_list    = list()
ref_list    = [list(), list()]
exact_match = 0
f1          = 0
for idx in range(len(predictions)):
    prediction  = predictions[idx][:-1].lower().strip(' .').split('.')[0].strip()
    ref1        = references1[idx][:-1].lower().strip(' .').strip()
    ref2        = references2[idx][:-1].lower().strip(' .').strip()
    ground_truths   = [ref1, ref2]
    hyp_list.append(prediction)
    ref_list[0].append(ref1)
    ref_list[1].append(ref2)
    exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1          += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

total_metrics_dict = compute_metrics(hyp_list, ref_list) 
total_metrics_dict['EM'] = exact_match / len(predictions)
total_metrics_dict['f1'] = f1 / len(predictions)



# Save the scores
print('###### Result of {} ######'.format(args.result_file))
# print(metrics_dict)
print(json.dumps(total_metrics_dict, indent=4))
with open(res_file, 'w+') as fp:
    print(json.dumps(total_metrics_dict, indent=4), file=fp)


