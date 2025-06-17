#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import datasets
from Levenshtein import distance as levenshtein_distance


# In[2]:


def calculate_iou(box_a, box_b):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_a (list): [x1, y1, x2, y2] for the first box.
        box_b (list): [x1, y1, x2, y2] for the second box.

    Returns:
        float: The IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersection_area

    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area


# In[7]:


def calculate_evidence_f1(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculates the Evidence F1 score based on bounding box matching. 

    Args:
        pred_boxes (list): A list of predicted bounding boxes.
        gt_boxes (list): A list of ground truth bounding boxes.
        iou_threshold (float): The threshold for a match. 

    Returns:
        float: The Evidence F1 score.
    """
    if not pred_boxes and not gt_boxes:
        return 1.0  # Perfect score if both are empty
    if not pred_boxes or not gt_boxes:
        return 0.0  # No overlap if one is empty

    true_positives = 0
    matched_gt_indices = set()

    # Perform one-to-one matching based on IoU 
    for p_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, g_box in enumerate(gt_boxes):
            if i in matched_gt_indices:
                continue
            iou = calculate_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold: # 
            true_positives += 1
            matched_gt_indices.add(best_gt_idx)

    precision = true_positives / len(pred_boxes) if pred_boxes else 0.0 # 
    recall = true_positives / len(gt_boxes) if gt_boxes else 0.0 # 
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall) # 
    return f1


# In[4]:


def calculate_anls(pred_str, gt_str, threshold=0.5):
    """
    Calculates Averaged Normalized Levenshtein Similarity (ANLS). 

    Args:
        pred_str (str): The predicted string.
        gt_str (str): The ground truth string.
        threshold (float): The threshold to discount large edit distances. 

    Returns:
        float: The ANLS score.
    """
    max_len = max(len(pred_str), len(gt_str))
    if max_len == 0:
        return 1.0
        
    nl_dist = levenshtein_distance(pred_str, gt_str) / max_len
    
    if nl_dist < threshold:
        return 1.0 - nl_dist
    else:
        return 0.0 # 


# In[5]:


def calculate_deviation_score(pred_val, gt_val):
    """
    Calculates the deviation score for numeric answers. 

    Args:
        pred_val (float): The predicted numeric value.
        gt_val (float): The ground truth numeric value.

    Returns:
        float: The Deviation Score.
    """
    if abs(pred_val - gt_val) > abs(gt_val):
        return 0.0 # 
    if gt_val == 0:
        return 1.0 if pred_val == 0 else 0.0
        
    return 1.0 - (abs(pred_val - gt_val) / abs(gt_val)) # 


# In[17]:


def calculate_string_component_f1(pred_components, gt_components):
    """Calculates F1 score for lists of strings."""
    if not pred_components and not gt_components: return 1.0
    if not pred_components or not gt_components: return 0.0
    pred_set = set(str(p) for p in pred_components)
    gt_set = set(str(g) for g in gt_components)
    true_positives = len(pred_set.intersection(gt_set))
    precision = true_positives / len(pred_set); recall = true_positives / len(gt_set)
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)


# In[18]:


def calculate_numeric_component_score(pred_components, gt_components):
    """Calculates a soft F1-style score for lists of numeric components."""
    if not gt_components:
        return 1.0 if not pred_components else 0.0
    
    try:
        pred_nums = [float(p) for p in pred_components]
    except (ValueError, TypeError):
        return 0.0 # Prediction contains non-numeric items

    if not pred_nums:
        return 0.0

    matched_pred_indices = set()
    total_match_score = 0.0

    # Greedily match each ground truth number to the best available predicted number
    for gt_num in gt_components:
        best_score, best_pred_idx = -1.0, -1
        for i, pred_num in enumerate(pred_nums):
            if i in matched_pred_indices: continue
            dev_score = calculate_deviation_score(pred_num, gt_num)
            anls_score = calculate_anls(str(pred_num), str(gt_num))
            pair_score = np.sqrt(dev_score**2 + anls_score**2) / np.sqrt(2)
            if pair_score > best_score:
                best_score, best_pred_idx = pair_score, i
        
        if best_pred_idx != -1:
            total_match_score += best_score
            matched_pred_indices.add(best_pred_idx)
            
    score_recall = total_match_score / len(gt_components)
    score_precision = total_match_score / len(pred_nums)

    if score_precision + score_recall == 0:
        return 0.0
    return 2 * (score_precision * score_recall) / (score_precision + score_recall)


# In[9]:


def gather_all_bboxes(data_dict):
    """Helper function to collect all bounding boxes from a dictionary."""
    all_boxes = []
    for key, value in data_dict.items():
        if key.endswith('_bbox') and value: all_boxes.append(value)
        elif key.endswith('_bboxes') and value: all_boxes.extend(value)
    return all_boxes


# In[39]:


def evaluate_sample(prediction, ground_truth):
    """Evaluates a prediction against its ground truth with nuanced component scoring."""
    pred_boxes = gather_all_bboxes(prediction)
    gt_boxes = gather_all_bboxes(ground_truth)
    evidence_score = calculate_evidence_f1(pred_boxes, gt_boxes)
    
    answer_scores = []
    
    # Score the final answer
    if 'answer' in ground_truth:
        pred_ans, gt_ans = prediction.get('answer'), ground_truth['answer']
        if isinstance(gt_ans, (int, float)):
            # try:
                pred_num = float(pred_ans)
                dev_score = calculate_deviation_score(pred_num, gt_ans)
                anls_score = calculate_anls(str(pred_ans), str(gt_ans))
                score = np.sqrt(dev_score**2 + anls_score**2) / np.sqrt(2)
            # except (ValueError, TypeError): score = 0.0
        else: score = calculate_anls(pred_ans, gt_ans)
        answer_scores.append(score)

    # Score the individual components
    if 'individual_answers' in ground_truth:
        gt_components = ground_truth['individual_answers']
        pred_components = prediction.get('individual_answers', [])
        # --- UPDATED LOGIC ---
        # Use numeric scoring if components are numbers, otherwise use string F1
        if gt_components and isinstance(gt_components[0], (int, float)):
            score = calculate_numeric_component_score(pred_components, gt_components)
        else:
            score = calculate_string_component_f1(pred_components, gt_components)
        answer_scores.append(score)
        
    # Score the answer key
    if 'answer_key' in ground_truth:
        score = calculate_anls(prediction.get('answer_key', ''), ground_truth['answer_key'])
        answer_scores.append(score)
    final_answer_score = np.mean(answer_scores) if answer_scores else 0.0
    total_score = evidence_score * final_answer_score
    
    return { "EvidenceF1": evidence_score, "AnswerScore_avg": final_answer_score, "TotalScore": total_score }
