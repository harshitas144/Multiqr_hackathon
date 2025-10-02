#!/usr/bin/env python3
"""
evaluate.py - Evaluation Script for Multi-QR Detection
Usage:
    python evaluate.py --predictions outputs/submission_detection_1.json --ground_truth data/annotations.json
    python evaluate.py --predictions outputs/submission_decoding_2.json --ground_truth data/annotations.json --stage 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Multi-QR Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth JSON file')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1,
                        help='Evaluation stage (1=detection, 2=decoding)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Load predictions and ground truth
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loading ground truth from: {args.ground_truth}")
    with open(args.ground_truth, 'r') as f:
        ground_truth = json.load(f)
    
    print(f"\nEvaluating {len(predictions)} predictions against {len(ground_truth)} ground truth images...")
    
    # Stage 1: Detection evaluation
    detection_metrics = evaluate_detection(predictions, ground_truth, args.iou_threshold)
    
    # Stage 2: Decoding evaluation (if applicable)
    decoding_metrics = None
    if args.stage == 2:
        has_decoded = any(
            'value' in qr 
            for pred in predictions 
            for qr in pred.get('qrs', [])
        )
        
        if has_decoded:
            decoding_metrics = evaluate_decoding(predictions, ground_truth, args.iou_threshold)
        else:
            print("\nWarning: Predictions do not contain decoded values. Skipping decoding evaluation.")
    
    # Print report
    print_evaluation_report(detection_metrics, decoding_metrics)
    
    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'detection': detection_metrics,
            'config': {
                'iou_threshold': args.iou_threshold,
                'stage': args.stage
            }
        }
        
        if decoding_metrics:
            results['decoding'] = decoding_metrics
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_path}")


def evaluate_detection(predictions: List[Dict], 
                       ground_truth: List[Dict], 
                       iou_threshold: float = 0.5) -> Dict:
    """Evaluate detection performance"""
    
    # Create lookup dictionaries
    gt_dict = {item['image_id']: item for item in ground_truth}
    pred_dict = {item['image_id']: item for item in predictions}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_image_results = []
    
    # Evaluate each image
    for image_id in gt_dict.keys():
        gt_boxes = [qr['bbox'] for qr in gt_dict[image_id]['qrs']]
        
        if image_id not in pred_dict:
            # All ground truth boxes are false negatives
            total_fn += len(gt_boxes)
            per_image_results.append({
                'image_id': image_id,
                'tp': 0,
                'fp': 0,
                'fn': len(gt_boxes),
                'precision': 0.0,
                'recall': 0.0
            })
            continue
        
        pred_boxes = [qr['bbox'] for qr in pred_dict[image_id]['qrs']]
        
        # Match predictions to ground truth
        matched_gt = set()
        image_tp = 0
        image_fp = 0
        
        for pred_box in pred_boxes:
            matched = False
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                image_tp += 1
                matched_gt.add(best_gt_idx)
                matched = True
            else:
                image_fp += 1
        
        image_fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += image_tp
        total_fp += image_fp
        total_fn += image_fn
        
        # Calculate per-image metrics
        img_precision = image_tp / (image_tp + image_fp) if (image_tp + image_fp) > 0 else 0
        img_recall = image_tp / (image_tp + image_fn) if (image_tp + image_fn) > 0 else 0
        
        per_image_results.append({
            'image_id': image_id,
            'tp': image_tp,
            'fp': image_fp,
            'fn': image_fn,
            'precision': img_precision,
            'recall': img_recall,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes)
        })
    
    # Check for predictions without ground truth
    for image_id in pred_dict.keys():
        if image_id not in gt_dict:
            pred_boxes = pred_dict[image_id]['qrs']
            total_fp += len(pred_boxes)
            per_image_results.append({
                'image_id': image_id,
                'tp': 0,
                'fp': len(pred_boxes),
                'fn': 0,
                'precision': 0.0,
                'recall': 0.0,
                'warning': 'No ground truth for this image'
            })
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'overall': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou_threshold': iou_threshold
        },
        'per_image': per_image_results
    }


def evaluate_decoding(predictions: List[Dict], 
                      ground_truth: List[Dict],
                      iou_threshold: float = 0.5) -> Dict:
    """Evaluate decoding and classification performance"""
    
    gt_dict = {item['image_id']: item for item in ground_truth}
    pred_dict = {item['image_id']: item for item in predictions}
    
    total_matches = 0
    correct_values = 0
    correct_types = 0
    correct_both = 0
    
    per_image_results = []
    
    for image_id in gt_dict.keys():
        if image_id not in pred_dict:
            continue
        
        gt_qrs = gt_dict[image_id]['qrs']
        pred_qrs = pred_dict[image_id]['qrs']
        
        image_matches = 0
        image_correct_values = 0
        image_correct_types = 0
        image_correct_both = 0
        
        # Match QRs by bounding box
        for gt_qr in gt_qrs:
            if 'value' not in gt_qr and 'data' not in gt_qr:
                continue
            
            gt_box = gt_qr['bbox']
            gt_value = gt_qr.get('value') or gt_qr.get('data', '')
            gt_type = gt_qr.get('type', 'unknown')
            
            # Find matching prediction
            best_match = None
            best_iou = 0
            
            for pred_qr in pred_qrs:
                if 'value' not in pred_qr and 'data' not in pred_qr:
                    continue
                
                pred_box = pred_qr['bbox']
                iou = calculate_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_qr
            
            if best_match and best_iou >= iou_threshold:
                image_matches += 1
                total_matches += 1
                
                pred_value = best_match.get('value') or best_match.get('data', '')
                pred_type = best_match.get('type', 'unknown')
                
                # Check value accuracy
                if pred_value == gt_value:
                    image_correct_values += 1
                    correct_values += 1
                
                # Check type accuracy
                if pred_type == gt_type:
                    image_correct_types += 1
                    correct_types += 1
                
                # Check both
                if pred_value == gt_value and pred_type == gt_type:
                    image_correct_both += 1
                    correct_both += 1
        
        per_image_results.append({
            'image_id': image_id,
            'matches': image_matches,
            'correct_values': image_correct_values,
            'correct_types': image_correct_types,
            'correct_both': image_correct_both,
            'value_accuracy': image_correct_values / image_matches if image_matches > 0 else 0,
            'type_accuracy': image_correct_types / image_matches if image_matches > 0 else 0
        })
    
    # Calculate overall metrics
    value_accuracy = correct_values / total_matches if total_matches > 0 else 0
    type_accuracy = correct_types / total_matches if total_matches > 0 else 0
    full_accuracy = correct_both / total_matches if total_matches > 0 else 0
    
    return {
        'overall': {
            'total_matches': total_matches,
            'correct_values': correct_values,
            'correct_types': correct_types,
            'correct_both': correct_both,
            'value_accuracy': value_accuracy,
            'type_accuracy': type_accuracy,
            'full_accuracy': full_accuracy
        },
        'per_image': per_image_results
    }


def print_evaluation_report(detection_metrics: Dict, decoding_metrics: Dict = None):
    """Print formatted evaluation report"""
    
    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}")
    
    # Detection metrics
    overall = detection_metrics['overall']
    print(f"\nDETECTION METRICS (Stage 1)")
    print("-" * 70)
    print(f"True Positives:      {overall['true_positives']:>6}")
    print(f"False Positives:     {overall['false_positives']:>6}")
    print(f"False Negatives:     {overall['false_negatives']:>6}")
    print(f"\nPrecision:           {overall['precision']:>6.2%}")
    print(f"Recall:              {overall['recall']:>6.2%}")
    print(f"F1 Score:            {overall['f1_score']:>6.2%}")
    print(f"IoU Threshold:       {overall['iou_threshold']:>6.2f}")
    
    # Decoding metrics (if available)
    if decoding_metrics:
        print(f"\nDECODING & CLASSIFICATION METRICS (Stage 2)")
        print("-" * 70)
        dec_overall = decoding_metrics['overall']
        print(f"Total Matches:       {dec_overall['total_matches']:>6}")
        print(f"Correct Values:      {dec_overall['correct_values']:>6}")
        print(f"Correct Types:       {dec_overall['correct_types']:>6}")
        print(f"Correct Both:        {dec_overall['correct_both']:>6}")
        print(f"\nValue Accuracy:      {dec_overall['value_accuracy']:>6.2%}")
        print(f"Type Accuracy:       {dec_overall['type_accuracy']:>6.2%}")
        print(f"Full Accuracy:       {dec_overall['full_accuracy']:>6.2%}")
    
    # Per-image summary
    print(f"\nPER-IMAGE SUMMARY")
    print("-" * 70)
    per_image = detection_metrics['per_image']
    
    # Find best and worst performing images
    images_with_scores = [
        (img['image_id'], img.get('precision', 0), img.get('recall', 0))
        for img in per_image
    ]
    images_with_scores.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
    
    print("\nTop 5 Performing Images:")
    for img_id, prec, rec in images_with_scores[:5]:
        print(f"  {img_id}: Precision={prec:.2%}, Recall={rec:.2%}")
    
    print("\nBottom 5 Performing Images:")
    for img_id, prec, rec in images_with_scores[-5:]:
        print(f"  {img_id}: Precision={prec:.2%}, Recall={rec:.2%}")
    
    print(f"\n{'='*70}\n")


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Handle different box formats
    if len(box1) == 4 and box1[2] < 2000:  # Likely [x, y, w, h]
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    else:  # [x_min, y_min, x_max, y_max]
        x1_min, y1_min, x1_max, y1_max = box1
    
    if len(box2) == 4 and box2[2] < 2000:  # Likely [x, y, w, h]
        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    else:  # [x_min, y_min, x_max, y_max]
        x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    main()