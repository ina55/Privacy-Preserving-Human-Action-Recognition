#!/usr/bin/env python3
"""
Face detection & recognition privacy analysis across datasets.
Compares face detection rates and embedding similarity: original vs cartoon.
Uses OpenCV Haar Cascade + ResNet embeddings.
"""
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FaceAnalyzer:
    def __init__(self, device='cpu'):
        self.device = device
        
        # OpenCV Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # ResNet50 for embedding extraction
        self.embedding_model = models.resnet50(pretrained=True).to(device)
        self.embedding_model.eval()
        # Use avgpool output as embedding (2048-dim)
        self.embedding_model = torch.nn.Sequential(
            *list(self.embedding_model.children())[:-1]
        )
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_faces(self, image):
        """Detect faces in image using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces  # List of (x, y, w, h)

    def extract_face_embedding(self, image, face_rect):
        """Extract embedding for detected face."""
        x, y, w, h = face_rect
        face_crop = image[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return None
        
        # Preprocess and extract embedding
        try:
            face_pil = transforms.ToPILImage()(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedding_model(face_tensor).squeeze().cpu().numpy()
            
            return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        except:
            return None

    def process_video(self, video_path, sample_rate=5, max_frames=None):
        """Extract faces from video at given sample rate."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        faces_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            if frame_idx % sample_rate == 0:
                detections = self.detect_faces(frame)
                embeddings = []
                
                for face_rect in detections:
                    emb = self.extract_face_embedding(frame, face_rect)
                    if emb is not None:
                        embeddings.append({
                            'embedding': emb.tolist()
                        })
                
                if embeddings:
                    faces_data.append({
                        'frame_idx': frame_idx,
                        'num_faces': len(embeddings),
                        'detections': embeddings
                    })
            
            frame_idx += 1
        
        cap.release()
        return faces_data if faces_data else None


def compare_embedding_similarity(emb1_list, emb2_list):
    """Compute average cosine similarity between embeddings."""
    if not emb1_list or not emb2_list:
        return None
    
    # Match closest pairs
    similarities = []
    for e1 in emb1_list:
        best_sim = max(
            np.dot(e1, e2) for e2 in emb2_list
        )
        similarities.append(best_sim)
    
    return float(np.mean(similarities)) if similarities else None


def analyze_dataset(dataset_root, dataset_name, original_folder, cartoon_folder, sample_rate=5, max_videos=None):
    """Analyze a full dataset."""
    
    analyzer = FaceAnalyzer(device='cpu')
    
    original_path = os.path.join(dataset_root, original_folder)
    cartoon_path = os.path.join(dataset_root, cartoon_folder)
    
    if not os.path.exists(original_path):
        print(f"[SKIP] {dataset_name}: original path not found: {original_path}")
        return None
    
    if not os.path.exists(cartoon_path):
        print(f"[SKIP] {dataset_name}: cartoon path not found: {cartoon_path}")
        return None
    
    # Find all video files
    video_exts = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.mpg'}
    original_videos = []
    
    for root, dirs, files in os.walk(original_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in video_exts:
                rel_path = os.path.relpath(os.path.join(root, f), original_path)
                original_videos.append(rel_path)
    
    if not original_videos:
        print(f"[SKIP] {dataset_name}: no videos found")
        return None
    
    print(f"\n[{dataset_name}] Found {len(original_videos)} videos, analyzing...")
    
    results = {
        'dataset': dataset_name,
        'total_videos': len(original_videos),
        'original_detection_rate': 0.0,
        'cartoon_detection_rate': 0.0,
        'avg_embedding_similarity': 0.0,
        'videos_with_faces_original': 0,
        'videos_with_faces_cartoon': 0,
        'total_faces_original': 0,
        'total_faces_cartoon': 0,
        'frame_samples_analyzed': 0,
        'video_details': []
    }
    
    total_faces_original = 0
    total_faces_cartoon = 0
    videos_with_faces_orig = 0
    videos_with_faces_cart = 0
    frame_samples = 0
    embedding_sims = []
    
    limit = min(len(original_videos), max_videos) if max_videos else len(original_videos)
    
    for vid_rel in tqdm(original_videos[:limit], desc=f"Analyzing {dataset_name}"):
        orig_file = os.path.join(original_path, vid_rel)
        cart_file = os.path.join(cartoon_path, vid_rel)
        
        if not os.path.exists(cart_file):
            continue
        
        # Process original
        orig_faces = analyzer.process_video(orig_file, sample_rate=sample_rate, max_frames=60)
        # Process cartoon
        cart_faces = analyzer.process_video(cart_file, sample_rate=sample_rate, max_frames=60)
        
        if orig_faces and cart_faces:
            orig_count = sum(f['num_faces'] for f in orig_faces)
            cart_count = sum(f['num_faces'] for f in cart_faces)
            
            total_faces_original += orig_count
            total_faces_cartoon += cart_count
            frame_samples += len(orig_faces)
            
            if orig_count > 0:
                videos_with_faces_orig += 1
            if cart_count > 0:
                videos_with_faces_cart += 1
            
            # Compute embedding similarity
            all_emb_orig = []
            all_emb_cart = []
            
            for frame_data in orig_faces:
                for det in frame_data['detections']:
                    all_emb_orig.append(np.array(det['embedding']))
            
            for frame_data in cart_faces:
                for det in frame_data['detections']:
                    all_emb_cart.append(np.array(det['embedding']))
            
            if all_emb_orig and all_emb_cart:
                sim = compare_embedding_similarity(all_emb_orig, all_emb_cart)
                if sim is not None:
                    embedding_sims.append(sim)
            
            results['video_details'].append({
                'video': vid_rel,
                'original_faces': orig_count,
                'cartoon_faces': cart_count,
                'face_detection_drop_%': 100.0 * (1.0 - cart_count / max(orig_count, 1))
            })
    
    if frame_samples > 0:
        results['original_detection_rate'] = total_faces_original / max(frame_samples, 1)
        results['cartoon_detection_rate'] = total_faces_cartoon / max(frame_samples, 1)
        results['avg_embedding_similarity'] = float(np.mean(embedding_sims)) if embedding_sims else 0.0
        results['videos_with_faces_original'] = videos_with_faces_orig
        results['videos_with_faces_cartoon'] = videos_with_faces_cart
        results['total_faces_original'] = total_faces_original
        results['total_faces_cartoon'] = total_faces_cartoon
        results['frame_samples_analyzed'] = frame_samples
    
    return results


def main():
    datasets_root = '/path/to/project/datasets'
    
    # Dataset configs: (root, dataset_name, original_folder, cartoon_folder, max_videos)
    configs = [
        (os.path.join(datasets_root, 'upb'), 'UPB', 'DatasetHAR', os.path.join('upb_cartoon', 'DatasetHAR'), None),  # All
        (os.path.join(datasets_root, 'hmdb51'), 'HMDB51', 'hmdb51_1', 'hmdb51_cartoon', None),  # ALL
        (os.path.join(datasets_root, 'ucf101'), 'UCF101', 'ucf101_1', 'ucf101_cartoon', None),  # ALL
    ]
    
    all_results = []
    
    for dataset_path, ds_name, orig_folder, cart_folder, max_vids in configs:
        result = analyze_dataset(
            dataset_path,
            ds_name,
            orig_folder,
            cart_folder,
            sample_rate=5,
            max_videos=max_vids  # Full for UPB, ALL for HMDB51/UCF101
        )
        if result:
            all_results.append(result)
    
    # Save results
    output_dir = '/path/to/project/work_dirs/privacy_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'face_detection_privacy_report.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("FACE DETECTION PRIVACY ANALYSIS SUMMARY")
    print("="*80)
    
    for result in all_results:
        if result['frame_samples_analyzed'] == 0:
            continue
            
        print(f"\n{result['dataset']}:")
        print(f"  Total videos: {result['total_videos']}")
        print(f"  Frame samples analyzed: {result['frame_samples_analyzed']}")
        print(f"  Faces detected (original): {result['total_faces_original']} total | {result['original_detection_rate']:.3f} avg/frame")
        print(f"  Faces detected (cartoon): {result['total_faces_cartoon']} total | {result['cartoon_detection_rate']:.3f} avg/frame")
        
        drop_pct = 100.0 * (1.0 - result['cartoon_detection_rate'] / max(result['original_detection_rate'], 0.0001))
        print(f"  Face detection drop (cartoon vs original): {drop_pct:.1f}%")
        
        print(f"  Videos with detected faces:")
        print(f"    Original: {result['videos_with_faces_original']}")
        print(f"    Cartoon: {result['videos_with_faces_cartoon']}")
        
        print(f"  Face identity preservation (embedding similarity): {result['avg_embedding_similarity']:.4f}")
        print(f"    (1.0 = perfect preservation, 0.0 = completely different)")
    
    print(f"\n✓ Detailed results saved to: {output_dir}/face_detection_privacy_report.json")
    return all_results


if __name__ == '__main__':
    main()
