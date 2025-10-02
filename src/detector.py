"""
detector.py - OPTIMIZED Multi-Method QR Code Detector
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from pyzbar import pyzbar
import warnings

warnings.filterwarnings('ignore')


class ImprovedQRDetector:
    """Optimized QR detector - faster processing"""
    
    def __init__(self):
        self.opencv_detector = cv2.QRCodeDetector()
        
        try:
            self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
            self.has_wechat = True
        except Exception:
            self.wechat_detector = None
            self.has_wechat = False
    
    def preprocess_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate only 2 most effective variants"""
        variants = []
        variants.append(('original', image))
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Only CLAHE - most effective
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(('clahe', clahe_img))
        
        return variants
    
    def detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect using OpenCV QRCodeDetector"""
        detections = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        try:
            success, points = self.opencv_detector.detect(gray)
            if success and points is not None:
                for point_set in points:
                    x_coords = point_set[:, 0]
                    y_coords = point_set[:, 1]
                    x_min = int(np.min(x_coords))
                    y_min = int(np.min(y_coords))
                    x_max = int(np.max(x_coords))
                    y_max = int(np.max(y_coords))
                    
                    detections.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'confidence': 0.95,
                        'method': 'opencv'
                    })
        except Exception:
            pass
        
        return detections
    
    def detect_wechat(self, image: np.ndarray) -> List[Dict]:
        """Detect using WeChat QR detector"""
        detections = []
        
        if not self.has_wechat:
            return detections
        
        try:
            if len(image.shape) == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            results, points = self.wechat_detector.detectAndDecode(image_bgr)
            
            if points is not None and len(points) > 0:
                for point_set in points:
                    if point_set is None or len(point_set) == 0:
                        continue
                    x_coords = point_set[:, 0]
                    y_coords = point_set[:, 1]
                    x_min = int(np.min(x_coords))
                    y_min = int(np.min(y_coords))
                    x_max = int(np.max(x_coords))
                    y_max = int(np.max(y_coords))
                    
                    detections.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'confidence': 0.98,
                        'method': 'wechat'
                    })
        except Exception:
            pass
        
        return detections
    
    def detect_pyzbar(self, image: np.ndarray) -> List[Dict]:
        """Detect using pyzbar"""
        detections = []
        
        try:
            codes = pyzbar.decode(image)
            for code in codes:
                if code.type == 'QRCODE':
                    x, y, w, h = code.rect
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 1.0,
                        'method': 'pyzbar',
                        'data': code.data.decode('utf-8', errors='ignore') if code.data else None
                    })
        except Exception:
            pass
        
        return detections
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
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
    
    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """NMS - optimized"""
        if len(detections) <= 1:
            return detections
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            detections = [
                det for det in detections
                if self.calculate_iou(det['bbox'], current['bbox']) < iou_threshold
            ]
        
        return keep
    
    def is_valid_qr_box(self, bbox: List[int], image_width: int, image_height: int) -> bool:
        """Quick validation"""
        x_min, y_min, x_max, y_max = bbox
        
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            return False
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        if box_w <= 0 or box_h <= 0:
            return False
        
        box_area = box_w * box_h
        image_area = image_width * image_height
        
        if box_area < 1000 or box_area > image_area * 0.25:
            return False
        
        aspect_ratio = box_w / box_h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False
        
        return True
    
    def decode_qr(self, image: np.ndarray, bbox: List[int]) -> str:
        """Fast decode - single attempt"""
        x_min, y_min, x_max, y_max = bbox
        
        padding = 10
        h, w = image.shape[:2]
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        roi = image[y_min:y_max, x_min:x_max]
        
        # Single pyzbar attempt
        codes = pyzbar.decode(roi)
        if codes:
            return codes[0].data.decode('utf-8', errors='ignore')
        
        return None
    
    def classify_qr(self, value: str) -> str:
        """Fast classification"""
        if not value or value == 'DECODE_FAILED':
            return 'unknown'
        
        value_upper = value.upper()
        
        type_keywords = {
            'manufacturer': ['MFR', 'MANUF', 'MAKER'],
            'batch': ['BATCH', 'LOT', 'B#'],
            'distributor': ['DIST', 'DISTR', 'SUPPLIER'],
            'regulator': ['REG', 'FDA', 'CERT'],
            'serial': ['SN', 'SERIAL', 'S/N'],
            'expiry': ['EXP', 'EXPIRY', 'DATE'],
        }
        
        for qr_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in value_upper:
                    return qr_type
        
        if any(char.isdigit() for char in value) and len(value) < 15:
            return 'serial'
        
        return 'unknown'
    
    def process_image(self, image_path: str, decode: bool = False) -> Dict:
        """OPTIMIZED detection pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            return {'image_id': Path(image_path).stem, 'qrs': []}
        
        h, w = image.shape[:2]
        all_detections = []
        
        # Only 2 variants instead of 8
        variants = self.preprocess_variants(image)
        
        for variant_name, variant_img in variants:
            all_detections.extend(self.detect_opencv(variant_img))
            all_detections.extend(self.detect_wechat(variant_img))
            all_detections.extend(self.detect_pyzbar(variant_img))
        
        # Quick filter
        valid_detections = [
            det for det in all_detections 
            if self.is_valid_qr_box(det['bbox'], w, h)
        ]
        
        # NMS
        final_detections = self.non_max_suppression(valid_detections, iou_threshold=0.4)
        
        # High confidence filter
        final_detections = [det for det in final_detections if det['confidence'] >= 0.90]
        
        # Format output
        qrs = []
        for det in final_detections:
            qr_info = {'bbox': det['bbox']}
            
            if decode:
                value = det.get('data')
                if not value:
                    value = self.decode_qr(image, det['bbox'])
                
                if not value:
                    value = "DECODE_FAILED"
                
                qr_info['value'] = value
                qr_info['type'] = self.classify_qr(value)
            
            qrs.append(qr_info)
        
        return {
            'image_id': Path(image_path).stem,
            'qrs': qrs
        }
       
    