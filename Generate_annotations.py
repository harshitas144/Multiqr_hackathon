import cv2
import json
import os
from pathlib import Path
from pyzbar import pyzbar
import numpy as np

def detect_qr_codes_pyzbar(image_path):
    """
    Detect QR codes using pyzbar (fast and accurate)
    Returns: list of dicts with bbox in [x_min, y_min, x_max, y_max] format
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    # Detect QR codes
    qr_codes = pyzbar.decode(image)
    
    results = []
    for qr in qr_codes:
        # Get bounding box coordinates
        x, y, w, h = qr.rect
        
        # Convert to [x_min, y_min, x_max, y_max] format
        bbox = [x, y, x + w, y + h]
        
        # For Stage 1 (detection only)
        result = {"bbox": bbox}
        
        # For Stage 2 (with decoding)
        # Uncomment below line if you want to include decoded values
        # result["value"] = qr.data.decode('utf-8')
        
        results.append(result)
    
    return results


def detect_qr_codes_opencv(image_path):
    """
    Detect QR codes using OpenCV's QRCodeDetector (backup method)
    Returns: list of dicts with bbox in [x_min, y_min, x_max, y_max] format
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    
    detector = cv2.QRCodeDetector()
    
    # Detect and decode
    data, points, _ = detector.detectAndDecode(image)
    
    results = []
    if points is not None:
        for point_set in points:
            # Get bounding box from points
            x_coords = point_set[:, 0]
            y_coords = point_set[:, 1]
            
            x_min = int(np.min(x_coords))
            y_min = int(np.min(y_coords))
            x_max = int(np.max(x_coords))
            y_max = int(np.max(y_coords))
            
            bbox = [x_min, y_min, x_max, y_max]
            
            result = {"bbox": bbox}
            
            # For Stage 2 (with decoding)
            # if data:
            #     result["value"] = data
            
            results.append(result)
    
    return results


def detect_qr_codes_wechat(image_path):
    """
    Detect QR codes using WeChat's QR detector (most robust)
    Download model from: https://github.com/WeChatCV/opencv_3rdparty
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        # Initialize WeChat QR Code Detector
        # Download these files first:
        # detect.prototxt and detect.caffemodel
        detector = cv2.wechat_qrcode_WeChatQRCode(
            "detect.prototxt",
            "detect.caffemodel",
            "sr.prototxt", 
            "sr.caffemodel"
        )
        
        data, points = detector.detectAndDecode(image)
        
        results = []
        if len(points) > 0:
            for i, point_set in enumerate(points):
                x_coords = point_set[:, 0]
                y_coords = point_set[:, 1]
                
                x_min = int(np.min(x_coords))
                y_min = int(np.min(y_coords))
                x_max = int(np.max(x_coords))
                y_max = int(np.max(y_coords))
                
                bbox = [x_min, y_min, x_max, y_max]
                
                result = {"bbox": bbox}
                
                # For Stage 2
                if i < len(data) and data[i]:
                    result["value"] = data[i]
                
                results.append(result)
        
        return results
    except:
        return []


def process_dataset(image_folder, output_json, method='pyzbar', stage=1):
    """
    Process all images in a folder and generate annotations
    
    Args:
        image_folder: Path to folder containing images
        output_json: Path to output JSON file
        method: 'pyzbar', 'opencv', or 'wechat'
        stage: 1 for detection only, 2 for detection + decoding
    """
    image_folder = Path(image_folder)
    annotations = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images")
    print(f"Using method: {method}")
    print(f"Stage: {stage}")
    print("-" * 50)
    
    # Select detection method
    if method == 'pyzbar':
        detect_func = detect_qr_codes_pyzbar
    elif method == 'opencv':
        detect_func = detect_qr_codes_opencv
    elif method == 'wechat':
        detect_func = detect_qr_codes_wechat
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        image_id = image_path.stem  # filename without extension
        
        print(f"[{idx}/{len(image_files)}] Processing {image_id}...", end=' ')
        
        # Detect QR codes
        qr_codes = detect_func(image_path)
        
        # For Stage 1, remove 'value' field if present
        if stage == 1:
            qr_codes = [{"bbox": qr["bbox"]} for qr in qr_codes]
        
        annotation = {
            "image_id": image_id,
            "qrs": qr_codes
        }
        
        annotations.append(annotation)
        print(f"✓ Found {len(qr_codes)} QR code(s)")
    
    # Save to JSON
    output_file = output_json if stage == 1 else output_json.replace('_1.json', '_2.json')
    with open(output_file, 'w') as f:
        json.dump(annotations, indent=2, fp=f)
    
    print("-" * 50)
    print(f"✓ Saved annotations to {output_file}")
    print(f"✓ Total images: {len(annotations)}")
    print(f"✓ Images with QR codes: {sum(1 for a in annotations if len(a['qrs']) > 0)}")
    print(f"✓ Total QR codes detected: {sum(len(a['qrs']) for a in annotations)}")
    
    return annotations


def visualize_annotations(image_path, annotations, output_path):
    """
    Draw bounding boxes on image for verification
    """
    image = cv2.imread(str(image_path))
    
    for qr in annotations:
        bbox = qr['bbox']
        x_min, y_min, x_max, y_max = bbox
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add label if value exists
        if 'value' in qr:
            label = qr['value'][:20]  # Truncate long values
            cv2.putText(image, label, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(str(output_path), image)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    
    # Path to your image folders
    TRAIN_FOLDER = "data/train_images"  # Change this
    TEST_FOLDER = "data/test_images"    # Change this
    
    # Output files
    TRAIN_OUTPUT = "data/train_annotations.json"
    TEST_OUTPUT = "data/test_annotations.json"
    
    # Detection method: 'pyzbar' (recommended), 'opencv', or 'wechat'
    METHOD = 'pyzbar'
    
    # Stage: 1 for detection only, 2 for detection + decoding
    STAGE = 1
    
    # ==================== INSTALLATION ====================
    print("=" * 50)
    print("INSTALLATION REQUIRED:")
    print("pip install opencv-python pyzbar pillow")
    print("=" * 50)
    print()
    
    # ==================== PROCESS DATASETS ====================
    
    # Process training images
    print("Processing TRAINING images...")
    train_annotations = process_dataset(TRAIN_FOLDER, TRAIN_OUTPUT, 
                                       method=METHOD, stage=STAGE)
    print()
    
    # Process test images
    print("Processing TEST images...")
    test_annotations = process_dataset(TEST_FOLDER, TEST_OUTPUT, 
                                      method=METHOD, stage=STAGE)
    print()
    
    # ==================== COMBINE (Optional) ====================
    # If you want a single file for submission
    all_annotations = train_annotations + test_annotations
    with open("submission_detection_1.json", 'w') as f:
        json.dump(all_annotations, indent=2, fp=f)
    
    print("=" * 50)
    print("✓ ALL DONE!")
    print(f"✓ Combined file saved to: submission_detection_1.json")
    print("=" * 50)
    
    # ==================== VISUALIZE (Optional) ====================
    # Uncomment to visualize first few results
    # if len(train_annotations) > 0:
    #     first_image = Path(TRAIN_FOLDER) / f"{train_annotations[0]['image_id']}.jpg"
    #     visualize_annotations(first_image, train_annotations[0]['qrs'], 
    #                          "visualization_sample.jpg")