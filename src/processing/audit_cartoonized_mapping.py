import os
import cv2
import numpy as np

def compare_frames(orig_path, cart_path, n=1):
    """Compare n-th frame of original and cartoonized video."""
    cap_orig = cv2.VideoCapture(orig_path)
    cap_cart = cv2.VideoCapture(cart_path)
    orig_frame, cart_frame = None, None
    for i in range(n+1):
        ret_o, f_o = cap_orig.read()
        ret_c, f_c = cap_cart.read()
        if i == n:
            orig_frame = f_o
            cart_frame = f_c
    cap_orig.release()
    cap_cart.release()
    if orig_frame is None or cart_frame is None:
        return False
    # Resize cartoonized frame to match original
    if orig_frame.shape != cart_frame.shape:
        cart_frame = cv2.resize(cart_frame, (orig_frame.shape[1], orig_frame.shape[0]))
    diff = np.mean(np.abs(orig_frame.astype(np.float32) - cart_frame.astype(np.float32)))
    return diff

def audit_mapping(orig_dir, cart_dir, ext_orig=".avi", ext_cart=".mp4", report_path="audit_report.txt"):
    mismatches = []
    for root, dirs, files in os.walk(orig_dir):
        for file in files:
            if not file.endswith(ext_orig):
                continue
            rel_path = os.path.relpath(os.path.join(root, file), orig_dir)
            cart_path = os.path.join(cart_dir, rel_path[:-len(ext_orig)] + ext_cart)
            if not os.path.exists(cart_path):
                mismatches.append((rel_path, "Missing cartoonized file"))
                continue
            diff = compare_frames(os.path.join(root, file), cart_path)
            if diff is False:
                mismatches.append((rel_path, "Frame read error"))
            elif diff < 5.0:
                mismatches.append((rel_path, f"Suspiciously similar (diff={diff:.2f})"))
    with open(report_path, "w") as f:
        for rel, reason in mismatches:
            f.write(f"{rel}: {reason}\n")
    print(f"Audit complete. See {report_path} for details.")

if __name__ == "__main__":
    orig_dir = "/path/to/project/datasets/hmdb51/hmdb51_1"
    cart_dir = "/path/to/project/datasets/hmdb51/hmdb51_cartoon_reencoded"
    audit_mapping(orig_dir, cart_dir)
