import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import argparse
from timm import create_model
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from collections import Counter

def attention_rollout(attns, discard_ratio=0.9):
    # attns: list of attention tensors [L x H x N x N]
    result = torch.eye(attns[0].size(-1), device=attns[0].device)
    for attn in attns:
        if attn.dim() == 4:  # [B,H,N,N]
            attn_heads = attn.mean(dim=1).squeeze(0)
        elif attn.dim() == 3:  # [H,N,N]
            attn_heads = attn.mean(dim=0)
        elif attn.dim() == 2:  # [N,N]
            attn_heads = attn
        else:
            raise ValueError(f"Unexpected attention shape {attn.shape}")
        attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
        # discard low attention
        flat = attn_heads.view(attn_heads.size(0), -1)
        _, idxs = torch.sort(flat, dim=-1)
        num = int(flat.size(-1) * discard_ratio)
        flat.scatter_(1, idxs[:, :num], 0)
        attn_heads = flat.view_as(attn_heads)
        result = attn_heads @ result
    return result  # [N x N]

# COCO-Stuff labels (subset of common categories)
COCO_LABELS = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def overlay_heatmap(image_bgr, heatmap):
    heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (heatmap[..., None] * 0.65).astype(np.float32)
    base = image_bgr.astype(np.float32)
    overlay = colored.astype(np.float32)
    blended = base * (1.0 - alpha) + overlay * alpha
    return blended.astype(np.uint8)


def colorize_heatmap(heatmap, output_size=None):
    if output_size is None:
        output_size = (heatmap.shape[1], heatmap.shape[0])
    resized = cv2.resize(heatmap, output_size)
    normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-6)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

def extract_attentions(model, x):
    # works for timm ViT-style models with blocks having attn.get_attention_map()
    attns = []
    hooks = []
    def hook_fn(module, inp, out):
        # Recompute attention weights inside the hook since timm Attention does not store them
        # inp[0]: [B, N, C]
        with torch.no_grad():
            x_in = inp[0]
            b, n, c = x_in.shape
            qkv = module.qkv(x_in)
            qkv = qkv.reshape(b, n, 3, module.num_heads, c // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attns.append(attn.detach().cpu())
    for blk in model.blocks:
        if hasattr(blk, "attn"):
            hooks.append(blk.attn.register_forward_hook(hook_fn))
    with torch.no_grad():
        _ = model(x)
    for h in hooks:
        h.remove()
    return attns  # list of [B,H,N,N]


def compute_frame_attention_heatmap(frame_bgr, model, device='cpu', input_size=224):
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    model.eval()
    img_tensor = tfm(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    attns = extract_attentions(model, img_tensor)
    attns = [a.squeeze(0).to(torch.float32) for a in attns]
    attns = torch.stack(attns, dim=0)
    rollout = attention_rollout(attns, discard_ratio=0.9)
    cls_to_patches = rollout[0]
    patch_scores = cls_to_patches[1:].numpy()

    patches_per_frame = (input_size // 16) * (input_size // 16)
    if patch_scores.size != patches_per_frame:
        raise ValueError(f"Unexpected token layout for frame with shape {frame_bgr.shape}")

    return patch_scores.reshape(input_size // 16, input_size // 16)

def get_attended_labels(frame, heatmap, det_model, device='cpu', top_k=3, debug=False):
    """Identify objects in high-attention regions using object detection."""
    # Run object detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])(frame_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        detections = det_model(det_input)[0]
    
    # Filter confident detections
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    
    conf_threshold = 0.3  # Lowered threshold to see more detections
    valid_mask = scores > conf_threshold
    boxes = boxes[valid_mask]
    labels = labels[valid_mask]
    scores = scores[valid_mask]
    
    if len(boxes) == 0:
        return []
    
    # Resize heatmap to frame size
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Calculate attention overlap for each detection
    attended_objects = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Get mean attention in this box
        box_attention = heatmap_resized[y1:y2, x1:x2].mean()
        label_name = COCO_LABELS[label - 1] if label <= len(COCO_LABELS) else f"unknown_{label}"
        attended_objects.append((label, label_name, score, box_attention, box))
        
        if debug:
            print(f"  Detected: {label_name} (conf={score:.2f}, attn={box_attention:.3f})")
    
    # Sort by attention score and get top labels
    attended_objects.sort(key=lambda x: x[3], reverse=True)
    
    seen_labels = set()
    top_labels = []
    for label_id, label_name, score, attn, box in attended_objects:
        if label_id not in seen_labels and label_name != 'background':
            top_labels.append(label_name)
            seen_labels.add(label_id)
        if len(top_labels) >= top_k:
            break
    
    return top_labels

def detect_attended_faces(frame, heatmap, attention_threshold=0.5):
    """Detect faces and check if they are in high-attention regions.
    Returns True if faces are found in attended regions."""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return False, []
    
    # Resize heatmap to frame size
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-6)
    
    # Check which faces have high attention
    attended_faces = []
    for (x, y, w_face, h_face) in faces:
        # Get mean attention in face region
        face_attention = heatmap_norm[y:y+h_face, x:x+w_face].mean()
        if face_attention > attention_threshold:
            attended_faces.append((x, y, w_face, h_face, face_attention))
    
    return len(attended_faces) > 0, attended_faces

def annotate_with_labels(image_bgr, labels):
    """Draw semantic labels on the image."""
    if not labels:
        return image_bgr
    
    img = image_bgr.copy()
    h, w = img.shape[:2]
    
    # Draw semi-transparent background box
    text = "Attending: " + ", ".join(labels)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position at top-left
    x, y = 10, 30
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

def visualize_video_attention(video_path, model, device='cpu', num_frames=16, input_size=224, save_path=None, frames_to_visualize=10, output_dir=None, det_model=None, annotate_labels=False):
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Could not determine video length, falling back to sequential read.")
        # Fallback: read sequentially
        frames = []
        while len(frames) < num_frames:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
    else:
        # Sample frames uniformly across the video
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        print(f"Sampling {num_frames} frames from {total_frames} total frames")
        print(f"Frame indices: {frame_indices.tolist()}")
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                frames.append(f)
            else:
                print(f"Warning: Could not read frame {idx}")
    
    cap.release()
    if len(frames) == 0:
        print("No frames decoded.")
        return

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    model.eval()

    # decide how many frames to visualize
    vis_count = min(len(frames), frames_to_visualize) if frames_to_visualize is not None else len(frames)

    # output directory and naming
    if output_dir is None:
        base = Path(video_path)
        out_dir = base.parent / f"attn_{base.stem}"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        stem = Path(video_path).stem
        suffix = ".jpg"
    else:
        target = Path(save_path)
        stem = target.stem if target.suffix else target.name
        suffix = target.suffix if target.suffix else ".jpg"

    for idx in range(vis_count):
        frame = frames[idx]
        img_tensor = tfm(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)  # [1,C,H,W]

        attns = extract_attentions(model, img_tensor)  # list of [B,H,N,N]
        attns = [a.squeeze(0).to(torch.float32) for a in attns]
        attns = torch.stack(attns, dim=0)
        rollout = attention_rollout(attns, discard_ratio=0.9)  # [N,N]
        cls_to_patches = rollout[0]
        patch_scores = cls_to_patches[1:].numpy()

        patches_per_frame = (input_size // 16) * (input_size // 16)
        if patch_scores.size != patches_per_frame:
            print(f"Unexpected token layout for frame {idx}. Skipping.")
            continue

        heat = patch_scores.reshape(input_size // 16, input_size // 16)
        vis = overlay_heatmap(frame, heat)
        
        # Annotate with semantic labels only if face is attended
        if annotate_labels:
            has_attended_face, face_info = detect_attended_faces(frame, heat, attention_threshold=0.3)
            
            if has_attended_face:
                print(f"\nFrame {idx}: Face detected in high-attention region")
                for face_idx, (x, y, w, h, attn) in enumerate(face_info):
                    print(f"  Face {face_idx + 1}: attention={attn:.3f}")
                
                # Annotate with "Face" label
                vis = annotate_with_labels(vis, ["Face"])
            else:
                print(f"\nFrame {idx}: No face in high-attention region (skipping annotation)")

        out_path = out_dir / f"{stem}_{idx:02d}{suffix}"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved attention visualization: {out_path}")

def load_model(ckpt_path, arch='vit_base_patch16_224', num_classes=400, device='cpu'):
    # This uses timm’s image ViT; checkpoint mismatches are expected.
    model = create_model(arch, pretrained=True, num_classes=num_classes)
    return model.to(device).eval()
def load_detection_model(device='cpu'):
    """Load Faster R-CNN object detection model."""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device).eval()
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to an mp4")
    parser.add_argument("--ckpt", default="./data/VideoMAEv2_data/vit_b_k710_dl_from_giant.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--frames_to_visualize", type=int, default=10, help="Number of frames to save (None for all decoded)")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to store all attention jpgs (default: attn_<video_stem> next to video)")
    parser.add_argument("--annotate_labels", action="store_true", help="Annotate only when face is in high-attention region")
    args = parser.parse_args()

    model = load_model(args.ckpt, device=args.device)
    
    visualize_video_attention(
        video_path=args.video,
        model=model,
        device=args.device,
        num_frames=args.num_frames,
        input_size=args.input_size,
        frames_to_visualize=args.frames_to_visualize,
        output_dir=args.output_dir,
        det_model=None,
        annotate_labels=args.annotate_labels,
    )
    