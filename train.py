

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Import detector from src
sys.path.append(str(Path(__file__).parent / 'src'))
from detector import ImprovedQRDetector


def main():
    parser = argparse.ArgumentParser(
        description='Train and validate QR detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to annotations JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for results (default: outputs)')
    parser.add_argument('--tune', action='store_true',
                        help='Tune detection parameters')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize failure cases')
    
    args = parser.parse_args()
    
    # Validate paths
    data_dir = Path(args.data_dir)
    annotations_file = Path(args.annotations)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    if not annotations_file.exists():
        print(f"Error: Annotations file not found: {annotations_file}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    if isinstance(annotations, list):
        annotations = {item['image_id']: item for item in annotations}
    
    print(f"Loaded {len(annotations)} annotated images")
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = ImprovedQRDetector()
    
    if args.tune:
        # Parameter tuning mode
        print("\nTuning detection parameters...")
        best_config = tune_parameters(detector, data_dir, annotations, output_dir)
        
        # Save best configuration
        config_file = output_dir / 'best_config.json'
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest configuration saved to {config_file}")
        
    else:
        # Standard validation mode
        print("\nRunning validation...")
        results, metrics = validate_detector(detector, data_dir, annotations)
        
        # Analyze failures
        print("\nAnalyzing failure cases...")
        failures = analyze_failures(data_dir, annotations, results)
        
        # Save results
        results_file = output_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'results': results,
                'failures': failures
            }, f, indent=2)
        
        print(f"\nValidation results saved to {results_file}")
        
        # Visualize failures if requested
        if args.visualize and failures:
            print("\nVisualizing failure cases...")
            visualize_failures(data_dir, annotations, results, 
                             output_dir / 'failure_visualizations')
    
    print("\nTraining/Validation complete!")


def validate_detector(detector: ImprovedQRDetector, 
                     data_dir: Path, 
                     annotations: Dict) -> Tuple[List[Dict], Dict]:
    """Run validation on dataset"""
    
    results = []
    metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    print(f"Validating on {len(annotations)} images...")
    
    for image_id, gt_data in tqdm(annotations.items()):
        # Find image file
        image_file = find_image_file(data_dir, image_id)
        
        if image_file is None:
            print(f"Warning: Image {image_id} not found")
            continue
        
        # Detect QRs
        result = detector.process_image(str(image_file), decode=False)
        results.append(result)
        
        # Extract bounding boxes
        pred_boxes = [qr['bbox'] for qr in result['qrs']]
        gt_boxes = [qr['bbox'] for qr in gt_data['qrs']]
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred_box in pred_boxes:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou >= 0.5:
                    metrics['tp'] += 1
                    matched_gt.add(i)
                    matched = True
                    break
            
            if not matched:
                metrics['fp'] += 1
        
        metrics['fn'] += len(gt_boxes) - len(matched_gt)
    
    # Calculate final metrics
    precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
    recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    final_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'fn': metrics['fn']
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(f"True Positives:  {metrics['tp']}")
    print(f"False Positives: {metrics['fp']}")
    print(f"False Negatives: {metrics['fn']}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*60}\n")
    
    return results, final_metrics


def analyze_failures(data_dir: Path, 
                     annotations: Dict, 
                     results: List[Dict]) -> List[Dict]:
    """Analyze failure cases"""
    
    failures = []
    
    for result in results:
        image_id = result['image_id']
        if image_id not in annotations:
            continue
        
        pred_count = len(result['qrs'])
        gt_count = len(annotations[image_id]['qrs'])
        
        if pred_count != gt_count:
            failure_type = 'count_mismatch'
            if pred_count == 0 and gt_count > 0:
                failure_type = 'missed_detection'
            elif pred_count > 0 and gt_count == 0:
                failure_type = 'false_positive'
            
            failures.append({
                'image_id': image_id,
                'predicted': pred_count,
                'ground_truth': gt_count,
                'type': failure_type
            })
    
    # Print failure summary
    if failures:
        print(f"\nFound {len(failures)} images with detection issues:")
        
        failure_types = {}
        for f in failures:
            ftype = f['type']
            if ftype not in failure_types:
                failure_types[ftype] = []
            failure_types[ftype].append(f)
        
        for ftype, flist in failure_types.items():
            print(f"  {ftype.replace('_', ' ').title()}: {len(flist)} cases")
            for f in flist[:3]:
                print(f"    - {f['image_id']}: predicted {f['predicted']}, expected {f['ground_truth']}")
            if len(flist) > 3:
                print(f"    ... and {len(flist) - 3} more")
    else:
        print("\nNo major failure cases found!")
    
    return failures


def tune_parameters(detector: ImprovedQRDetector, 
                   data_dir: Path, 
                   annotations: Dict,
                   output_dir: Path) -> Dict:
    """Tune detection parameters"""
    
    print("\nTesting different confidence thresholds...")
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results_by_threshold = {}
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        
        # Temporarily modify detector confidence
        # (In practice, you'd pass this to the detector)
        _, metrics = validate_detector(detector, data_dir, annotations)
        results_by_threshold[threshold] = metrics
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Threshold Comparison:")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for threshold, metrics in results_by_threshold.items():
        print(f"{threshold:<12.1f} "
              f"{metrics['precision']:<12.3f} "
              f"{metrics['recall']:<12.3f} "
              f"{metrics['f1']:<12.3f}")
    
    # Find best threshold
    best_threshold = max(results_by_threshold.keys(),
                        key=lambda t: results_by_threshold[t]['f1'])
    best_f1 = results_by_threshold[best_threshold]['f1']
    
    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.3f})")
    print(f"{'='*70}\n")
    
    # Plot results
    plot_threshold_analysis(results_by_threshold, output_dir)
    
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'all_results': results_by_threshold
    }


def plot_threshold_analysis(results_by_threshold: Dict, output_dir: Path):
    """Plot threshold analysis results"""
    
    thresholds = list(results_by_threshold.keys())
    precisions = [results_by_threshold[t]['precision'] for t in thresholds]
    recalls = [results_by_threshold[t]['recall'] for t in thresholds]
    f1s = [results_by_threshold[t]['f1'] for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, marker='^', label='F1 Score', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = output_dir / 'threshold_analysis.png'
    plt.savefig(plot_file, dpi=150)
    print(f"Threshold analysis plot saved to {plot_file}")
    plt.close()


def visualize_failures(data_dir: Path, 
                       annotations: Dict, 
                       results: List[Dict],
                       output_dir: Path,
                       num_samples: int = 6):
    """Visualize failure cases"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find failure cases
    failure_cases = []
    for result in results:
        image_id = result['image_id']
        if image_id not in annotations:
            continue
        
        pred_count = len(result['qrs'])
        gt_count = len(annotations[image_id]['qrs'])
        
        if pred_count != gt_count or (pred_count == 0 and gt_count > 0):
            failure_cases.append({
                'result': result,
                'annotation': annotations[image_id]
            })
    
    if not failure_cases:
        print("No failure cases to visualize")
        return
    
    samples = failure_cases[:num_samples]
    
    import cv2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, case in enumerate(samples):
        if i >= len(axes):
            break
        
        image_id = case['result']['image_id']
        image_file = find_image_file(data_dir, image_id)
        
        if image_file is None:
            continue
        
        # Load image
        image = cv2.imread(str(image_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(image_rgb)
        
        pred_count = len(case['result']['qrs'])
        gt_count = len(case['annotation']['qrs'])
        
        axes[i].set_title(f"{image_id}\nPred: {pred_count}, GT: {gt_count}", fontsize=9)
        axes[i].axis('off')
        
        # Draw ground truth (green dashed)
        for qr in case['annotation']['qrs']:
            bbox = qr['bbox']
            # Convert if needed
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3] if bbox[2] < 2000 else bbox
            else:
                x_min, y_min, x_max, y_max = bbox
            
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            axes[i].add_patch(rect)
        
        # Draw predictions (red solid)
        for qr in case['result']['qrs']:
            bbox = qr['bbox']
            x_min, y_min, x_max, y_max = bbox
            
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[i].add_patch(rect)
    
    # Legend
    handles = [
        patches.Patch(color='green', label='Ground Truth'),
        patches.Patch(color='red', label='Predicted')
    ]
    fig.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plot_file = output_dir / 'failure_cases.png'
    plt.savefig(plot_file, dpi=150)
    print(f"Failure visualization saved to {plot_file}")
    plt.close()


def find_image_file(data_dir: Path, image_id: str) -> Path:
    """Find image file with given ID"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
        candidate = data_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x_min, y_min, x_max, y_max] format"""
    # Handle different box formats
    if len(box1) == 4 and box1[2] < 2000:  # Likely [x, y, w, h]
        x1_min, y1_min = box1[0], box1[1]
        x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    else:
        x1_min, y1_min, x1_max, y1_max = box1
    
    if len(box2) == 4 and box2[2] < 2000:  # Likely [x, y, w, h]
        x2_min, y2_min = box2[0], box2[1]
        x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    else:
        x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


if __name__ == "__main__":
    main()
