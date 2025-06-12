import os
import json
import argparse

def compute_acc_F1(gt_lst, pred_lst):

    # Calculate accuracy
    correct = sum(1 for p, g in zip(pred_lst, gt_lst) if p == g)
    accuracy = correct / len(pred_lst)
    # print(correct, len(pred_lst))
    # import sys;sys.exit(0)

    # Calculate F1 score components
    true_pos = sum(1 for p, g in zip(pred_lst, gt_lst) if p == 1 and g == 1)
    false_pos = sum(1 for p, g in zip(pred_lst, gt_lst) if p == 1 and g == 0)
    false_neg = sum(1 for p, g in zip(pred_lst, gt_lst) if p == 0 and g == 1)
    
    # Calculate precision and recall
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def get_gt_lst(file_path):
    gt_lst, pred_lst = [], []
    correct_image = []
    with open(file_path, 'r') as f:
        for line in f:
            line = json.loads(line)

            answer = line['text'].lower().strip()
            if 'real' in answer:
                pred_lst.append(0)
            elif 'fake' in answer or 'computer-generated' in answer:
                pred_lst.append(1)
            else:
                print(line['image'])
                print(answer)
                continue

            if line['image'].startswith('./utils/DDVQA_images/c40/test/5'):
                gt_lst.append(0)
            else:
                gt_lst.append(1)

    return gt_lst, pred_lst

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path', type=str, default= "./outputs/DDVQA/DDVQA_det_c40.jsonl")
    args = parser.parse_args()

    gt_lst, pred_lst = get_gt_lst(args.predict_path)
    assert len(pred_lst) == len(gt_lst)
    compute_acc_F1(gt_lst, pred_lst)