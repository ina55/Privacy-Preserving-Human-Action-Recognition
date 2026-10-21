import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import torch
import numpy as np
import decord
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.modeling_finetune import vit_base_patch16_224, vit_giant_patch14_224
from dataset import video_transforms, volume_transforms

def load_label_map(label_map_file):
    labels = []
    with open(label_map_file, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels


def build_label_map_from_csv(csv_path):
    label_to_class = {}
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            label = int(parts[1])
            if rel_path.startswith("hmdb51_1/"):
                rel_path = rel_path[len("hmdb51_1/"):]
            class_name = rel_path.split('/')[0]
            if label not in label_to_class:
                label_to_class[label] = class_name

    if not label_to_class:
        raise RuntimeError(f"Could not infer class labels from {csv_path}")

    max_label = max(label_to_class.keys())
    labels = [label_to_class.get(i, f"class_{i}") for i in range(max_label + 1)]
    return labels


def resolve_video_path(hmdb51_dir, rel_path):
    rel_path_obj = Path(rel_path)
    full_path = hmdb51_dir / rel_path_obj
    if full_path.exists() and full_path.is_file() and not full_path.name.startswith("._"):
        return full_path

    class_dir = hmdb51_dir / rel_path_obj.parent
    if not class_dir.exists():
        return None

    stem = rel_path_obj.stem
    candidates = sorted(class_dir.glob(f"{stem}.*"))
    valid_exts = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.m4v'}
    for cand in candidates:
        if cand.is_file() and cand.suffix.lower() in valid_exts and not cand.name.startswith("._"):
            return cand

    return None

def anonymize_video(video_path, anonymization_script, output_dir):
    video_id = video_path.stem
    anonymized_path = output_dir / f"anonymized_{video_id}.mp4"
    
    # Check if already exists to save time
    if anonymized_path.exists():
        return anonymized_path
        
    try:
        result = subprocess.run([
            "python3", str(anonymization_script),
            str(video_path),
            str(output_dir)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and anonymized_path.exists():
            return anonymized_path
        return None
    except Exception as e:
        print(f"Anonymization error: {e}")
        return None

def load_model(checkpoint_path, model_type='vit_base', num_classes=51, device='cpu'):
    print(f"Loading {model_type} model...")
    if model_type == 'vit_giant':
        # Default initialization based on training script
        model = vit_giant_patch14_224(
            num_classes=num_classes,
            drop_path_rate=0.35, # From start_finetune_fixed.sh
        )
    else:
        model = vit_base_patch16_224(
            num_classes=num_classes,
            drop_path_rate=0.1,
        )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try different checkpoint keys
    checkpoint_model = None
    for model_key in ['model', 'module', 'model_state']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Load state_dict by model_key = {model_key}")
            break
            
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        print("Load state_dict directly from checkpoint")
        
    for old_key in list(checkpoint_model.keys()):
        if old_key.startswith('_orig_mod.'):
            new_key = old_key[10:]
            checkpoint_model[new_key] = checkpoint_model.pop(old_key)
            
    # Remove classification head if numb_classes do not match
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint due to size mismatch")
            del checkpoint_model[k]
            
    model.load_state_dict(checkpoint_model, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

def preprocess_video(video_path, num_frames=16, input_size=224):
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    
    if total_frames > num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        
    frames = vr.get_batch(indices).asnumpy()
    frames_pil = [Image.fromarray(frame) for frame in frames]
    
    transform = video_transforms.Compose([
        video_transforms.Resize(input_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(input_size, input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_tensor = transform(frames_pil)
    video_tensor = video_tensor.unsqueeze(0)
    return video_tensor

def run_inference_on_video(video_path, model, device='cpu'):
    try:
        # Determine input size from model
        input_size = model.patch_embed.img_size[0] if hasattr(model.patch_embed, 'img_size') else 224
        
        video_tensor = preprocess_video(video_path, input_size=input_size)
        video_tensor = video_tensor.to(device)
        
        with torch.no_grad():
            output = model(video_tensor)
            
            # Get Top 5 predictions
            probs = torch.nn.functional.softmax(output, dim=1)
            top5_prob, top5_indices = torch.topk(probs, k=5, dim=1)
            
            return {
                'top1_pred': top5_indices[0][0].item(),
                'top5_preds': top5_indices[0].tolist(),
                'probs': top5_prob[0].tolist()
            }
            
    except Exception as e:
        print(f"Inference error on {video_path}: {e}")
        return None

def compute_metrics(results):
    """Compute overall and per-class metrics."""
    metrics = {
        'overall': {'total': 0, 'top1_correct': 0, 'top5_correct': 0},
        'per_class': {}
    }
    
    for res in results:
        label = res['label']
        class_name = res['class_name']
        
        if class_name not in metrics['per_class']:
            metrics['per_class'][class_name] = {'total': 0, 'top1_correct': 0, 'top5_correct': 0}
            
        # Top-1 Correct
        is_top1 = res['top1_pred'] == label
        # Top-5 Correct
        is_top5 = label in res['top5_preds']
        
        metrics['overall']['total'] += 1
        metrics['per_class'][class_name]['total'] += 1
        
        if is_top1:
            metrics['overall']['top1_correct'] += 1
            metrics['per_class'][class_name]['top1_correct'] += 1
        if is_top5:
            metrics['overall']['top5_correct'] += 1
            metrics['per_class'][class_name]['top5_correct'] += 1
            
    # Calculate percentages
    if metrics['overall']['total'] > 0:
        metrics['overall']['top1_acc'] = (metrics['overall']['top1_correct'] / metrics['overall']['total']) * 100
        metrics['overall']['top5_acc'] = (metrics['overall']['top5_correct'] / metrics['overall']['total']) * 100
    
    for cls_name, data in metrics['per_class'].items():
        if data['total'] > 0:
            data['top1_acc'] = (data['top1_correct'] / data['total']) * 100
            data['top5_acc'] = (data['top5_correct'] / data['total']) * 100
            
    return metrics

def write_report(original_metrics, anonymized_metrics, label_names, output_dir, report_name, include_anonymized=True):
    report_path = output_dir / report_name
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        if include_anonymized:
            f.write("HMDB51 EVALUATION REPORT (ORIGINAL VS ANONYMIZED)\n")
        else:
            f.write("HMDB51 EVALUATION REPORT (ORIGINAL ONLY)\n")
        f.write("="*80 + "\n\n")
        
        # Overall Summary
        f.write("--- OVERALL SUMMARY ---\n")
        
        ob_total = original_metrics['overall']['total']
        ob_top1 = original_metrics['overall'].get('top1_acc', 0)
        ob_top5 = original_metrics['overall'].get('top5_acc', 0)
        
        f.write(f"Total Videos Evaluated: {ob_total}\n\n")
        
        f.write(f"Original Videos:\n")
        f.write(f"  Top-1 Accuracy: {ob_top1:.2f}% ({original_metrics['overall']['top1_correct']}/{ob_total})\n")
        f.write(f"  Top-5 Accuracy: {ob_top5:.2f}% ({original_metrics['overall']['top5_correct']}/{ob_total})\n\n")
        if include_anonymized:
            an_total = anonymized_metrics['overall']['total']
            an_top1 = anonymized_metrics['overall'].get('top1_acc', 0)
            an_top5 = anonymized_metrics['overall'].get('top5_acc', 0)

            f.write(f"Anonymized (Blurred) Videos:\n")
            f.write(f"  Top-1 Accuracy: {an_top1:.2f}% ({anonymized_metrics['overall']['top1_correct']}/{an_total})\n")
            f.write(f"  Top-5 Accuracy: {an_top5:.2f}% ({anonymized_metrics['overall']['top5_correct']}/{an_total})\n\n")

            f.write(f"Difference (Original - Anonymized):\n")
            f.write(f"  Top-1 Drop: {ob_top1 - an_top1:.2f}%\n")
            f.write(f"  Top-5 Drop: {ob_top5 - an_top5:.2f}%\n\n")
        
        # Per-Class Breakdown
        f.write("--- PER-CLASS BREAKDOWN ---\n")
        if include_anonymized:
            f.write(f"{'Class Name':<20} | {'Orig Top-1':<10} | {'Anon Top-1':<10} | {'Diff':<8} || {'Orig Top-5':<10} | {'Anon Top-5':<10} | {'Total':<5}\n")
            f.write("-" * 95 + "\n")
        else:
            f.write(f"{'Class Name':<20} | {'Top-1':<10} | {'Top-5':<10} | {'Total':<5}\n")
            f.write("-" * 60 + "\n")
        
        for class_name in label_names:
            orig_data = original_metrics['per_class'].get(class_name, {'total': 0, 'top1_acc': 0, 'top5_acc': 0})
            if orig_data['total'] == 0:
                continue

            orig_t1 = orig_data.get('top1_acc', 0)
            orig_t5 = orig_data.get('top5_acc', 0)

            if include_anonymized:
                anon_data = anonymized_metrics['per_class'].get(class_name, {'total': 0, 'top1_acc': 0, 'top5_acc': 0})
                anon_t1 = anon_data.get('top1_acc', 0)
                diff_t1 = orig_t1 - anon_t1
                anon_t5 = anon_data.get('top5_acc', 0)
                f.write(f"{class_name:<20} | {orig_t1:>9.2f}% | {anon_t1:>9.2f}% | {diff_t1:>7.2f}% || {orig_t5:>9.2f}% | {anon_t5:>9.2f}% | {orig_data['total']:>5}\n")
            else:
                f.write(f"{class_name:<20} | {orig_t1:>9.2f}% | {orig_t5:>9.2f}% | {orig_data['total']:>5}\n")
            
    print(f"\nReport successfully saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate HMDB51 local dataset")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--model', type=str, default='vit_giant', choices=['vit_base', 'vit_giant'], help="Model architecture")
    parser.add_argument('--limit', type=int, default=0, help="Limit number of videos to evaluate (for debugging)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for inference")
    parser.add_argument('--hmdb51-dir', type=str, default='', help="Path to HMDB51 dataset directory containing test.csv")
    parser.add_argument('--report-name', type=str, default='eval_report_low.txt', help="Report filename written inside the results directory")
    parser.add_argument('--results-name', type=str, default='eval_results_raw_low.json', help="Raw JSON results filename written inside the results directory")
    parser.add_argument('--anonymize-video', dest='anonymize_video', action='store_true', help="Run anonymization and compare original vs anonymized")
    parser.add_argument('--no-anonymize-video', dest='anonymize_video', action='store_false', help="Skip anonymization and evaluate only original videos")
    parser.set_defaults(anonymize_video=True)
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    workspace_root = base_dir.parents[1]
    hmdb51_dir = Path(args.hmdb51_dir) if args.hmdb51_dir else (workspace_root / "datasets/hmdb51/hmdb51_1")
    anonymization_script = base_dir / "input/anonymization/anonymization.py"
    anonymized_output_env = os.getenv("ANON_OUTPUT_DIR", "").strip()
    anonymized_output_dir = Path(anonymized_output_env) if anonymized_output_env else (workspace_root / "datasets/hmdb51/hmdb51_anonimized")
    results_dir = workspace_root / "datasets/hmdb51/hmdb51_results"

    if args.anonymize_video:
        anonymized_output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir: {hmdb51_dir}")
    print(f"Anonymization enabled: {args.anonymize_video}")
    
    # Load Label Map
    label_map_file = hmdb51_dir / "label_map.txt"
    train_csv = hmdb51_dir / "train.csv"
    test_csv = hmdb51_dir / "test.csv"

    if label_map_file.exists():
        label_names = load_label_map(label_map_file)
    elif train_csv.exists():
        label_names = build_label_map_from_csv(train_csv)
    elif test_csv.exists():
        label_names = build_label_map_from_csv(test_csv)
    else:
        raise FileNotFoundError(
            f"Could not find label_map.txt, train.csv, or test.csv in {hmdb51_dir}"
        )
    print(f"Loaded {len(label_names)} classes.")
    
    # Load test.csv
    videos_to_eval = []
    class_counts = {}
    
    with open(test_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # e.g hmdb51_1/brush_hair/Aussie_Brunette...mp4 0
                rel_path = parts[0]
                # Handle old paths prefixed with hmdb51_1/
                if rel_path.startswith("hmdb51_1/"):
                    rel_path = rel_path[len("hmdb51_1/"):]
                
                full_path = resolve_video_path(hmdb51_dir, rel_path)
                label = int(parts[1])
                class_name = label_names[label]
                
                if full_path is not None:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    videos_to_eval.append({
                        'path': full_path,
                        'label': label,
                        'class_name': class_name
                    })

    if args.limit > 0:
        videos_to_eval = videos_to_eval[:args.limit]
        
    print(f"Found {len(videos_to_eval)} videos to evaluate.")
    
    # Load Model
    model = load_model(args.checkpoint, model_type=args.model, num_classes=len(label_names), device=args.device)
    
    original_results = []
    cartoonized_results = []
    
    try:
        for idx, video_info in enumerate(videos_to_eval, 1):
            video_path = video_info['path']
            label = video_info['label']
            class_name = video_info['class_name']

            print(f"\n[{idx}/{len(videos_to_eval)}] Processing: {video_path.name} (Class: {class_name})")

            # 1. Original Inference
            res_orig = run_inference_on_video(video_path, model, args.device)
            if res_orig:
                res_orig['label'] = label
                res_orig['class_name'] = class_name
                original_results.append(res_orig)

                is_correct = res_orig['top1_pred'] == label
                print(f"  ✓ Original Top-1: {res_orig['top1_pred']} (Correct: {is_correct})")
            else:
                print(f"  ✗ Original inference failed")
                continue

            if not args.anonymize_video:
                continue

            # 2. Cartoonize and Evaluate
            cartoon_dir = Path("/path/to/project/datasets/hmdb51/hmdb51_cartoon") / class_name
            cartoon_dir.mkdir(parents=True, exist_ok=True)
            cartoonized_path = cartoon_dir / video_path.name

            if cartoonized_path.exists():
                print(f"  • Cartoonized video already exists: {cartoonized_path}")
            else:
                # Clean VideoToCartoon input/output
                video_input = Path("/tmp/VideoToCartoon/video_input")
                video_output = Path("/tmp/VideoToCartoon/video_output")
                for f in video_input.glob("*"): f.unlink()
                for f in video_output.glob("*"): f.unlink()
                # Copy input video
                import shutil
                shutil.copy2(video_path, video_input / video_path.name)
                # Run VideoToCartoon
                import subprocess
                try:
                    subprocess.run(["python3", "main.py", "--input", video_path.name], cwd="/tmp/VideoToCartoon", check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Cartoonization subprocess failed for {video_path.name}: {e}")
                    continue
                # Move cartoonized output
                cartoonized_temp = video_output / f"{video_path.stem}.avi"
                if cartoonized_temp.exists():
                    shutil.move(str(cartoonized_temp), str(cartoonized_path))
                else:
                    print(f"  ✗ Cartoonization failed for {video_path.name}")
                    continue

            # Evaluate cartoonized video
            res_cartoon = run_inference_on_video(cartoonized_path, model, args.device)
            if res_cartoon:
                res_cartoon['label'] = label
                res_cartoon['class_name'] = class_name
                cartoonized_results.append(res_cartoon)
                is_correct = res_cartoon['top1_pred'] == label
                print(f"  ✓ Cartoonized Top-1: {res_cartoon['top1_pred']} (Correct: {is_correct})")
            else:
                print(f"  ✗ Cartoonized inference failed")
                continue
    
    except KeyboardInterrupt:
        print("\n\n[!] Evaluation interrupted by user! Saving partial results...")

    finally:
        # Compute Metrics
        orig_metrics = compute_metrics(original_results)
        cartoon_metrics = compute_metrics(cartoonized_results)

        # Save Raw Results
        with open(results_dir / args.results_name, 'w') as f:
            json.dump({
                'original': original_results,
                'cartoonized': cartoonized_results,
                'metrics_original': orig_metrics,
                'metrics_cartoonized': cartoon_metrics
            }, f, indent=2)

        # Write Final Report (original only or original vs cartoonized)
        write_report(orig_metrics, cartoon_metrics, label_names, results_dir, args.report_name, include_anonymized=args.anonymize_video)

if __name__ == "__main__":
    main()
