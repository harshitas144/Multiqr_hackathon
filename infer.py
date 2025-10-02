


import argparse
import json
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent / 'src'))
from detector import ImprovedQRDetector


def main():
    parser = argparse.ArgumentParser(
        description='Multi-QR Code Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Detection only
  python infer.py --input data/test/ --output outputs/submission_detection_1.json
  
  # Stage 2: Detection + Decoding + Classification
  python infer.py --input data/test/ --output outputs/submission_decoding_2.json --decode
  
  # Process single image
  python infer.py --input data/test/img001.jpg --output result.json
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or folder path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--decode', action='store_true',
                        help='Enable QR decoding and classification (Stage 2)')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    print("Initializing Multi-QR Detection Pipeline...")
    detector = ImprovedQRDetector()
    

    if input_path.is_file():
        print(f"Processing single image: {input_path.name}")
        results = [detector.process_image(str(input_path), decode=args.decode)]
    elif input_path.is_dir():
        print(f"Processing folder: {input_path}")
        results = process_folder(detector, input_path, decode=args.decode)
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_summary(results, output_path, stage=2 if args.decode else 1)


def process_folder(detector, folder_path, decode=False):
    """Process all images in folder"""
    from tqdm import tqdm
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(folder_path.glob(f"*{ext}")))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"Warning: No images found in {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images")
    
    results = []
    for image_file in tqdm(image_files, desc="Processing"):
        try:
            result = detector.process_image(str(image_file), decode=decode)
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            # Add empty result to maintain consistency
            results.append({
                'image_id': image_file.stem,
                'qrs': []
            })
    
    return results


def print_summary(results, output_path, stage=1):
    """Print processing summary"""
    print(f"\n{'='*60}")
    print(f"Inference Complete - Stage {stage}")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    
    total_qrs = sum(len(r['qrs']) for r in results)
    images_with_qrs = sum(1 for r in results if len(r['qrs']) > 0)
    
    print(f"Images with QRs detected: {images_with_qrs}")
    print(f"Total QR codes detected: {total_qrs}")
    
    if total_qrs > 0:
        print(f"Average QRs per image: {total_qrs/len(results):.2f}")
    
    if stage == 2:
        decoded_count = sum(
            1 for r in results 
            for qr in r['qrs'] 
            if 'value' in qr and qr['value'] != 'DECODE_FAILED'
        )
        print(f"Successfully decoded: {decoded_count}")
        if total_qrs > 0:
            print(f"Decode success rate: {decoded_count/total_qrs*100:.1f}%")
    
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()