import argparse
import json
import os
import sys

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import input_data
from i3d import InceptionI3d


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a Grad-CAM strip for an RGB I3D checkpoint.')
    parser.add_argument('--video-path', required=True, help='Input video file path.')
    parser.add_argument('--num-classes', required=True, type=int, help='Number of classes in the checkpoint head.')
    parser.add_argument('--checkpoint-dir', help='Directory containing TensorFlow checkpoint files.')
    parser.add_argument('--checkpoint-path', help='Explicit checkpoint prefix, e.g. /path/to/i3d_rgb-9999')
    parser.add_argument('--reference-list', help='Optional train/test list used to map class ids back to class names.')
    parser.add_argument('--target-class', type=int, help='Optional explicit class index for Grad-CAM.')
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--sample-rate', type=int, default=2)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--display-frames', type=int, default=8, help='Number of frames to show in the output strip.')
    parser.add_argument('--overlay-alpha', type=float, default=0.45)
    parser.add_argument('--output-path', required=True, help='Output PNG path.')
    parser.add_argument('--metadata-path', help='Optional JSON metadata path.')
    parser.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    return parser.parse_args()


def resolve_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path
    if not args.checkpoint_dir:
        raise ValueError('Either --checkpoint-path or --checkpoint-dir must be provided.')
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    raise ValueError('No checkpoint found in {}'.format(args.checkpoint_dir))


def read_reference_list(path):
    label_to_class = {}
    if not path:
        return label_to_class
    with open(path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            video_path, label = line.rsplit(' ', 1)
            label_to_class.setdefault(int(label), os.path.basename(os.path.dirname(video_path)))
    return label_to_class


def get_center_clip(video_path, num_frames, sample_rate, crop_size):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError('Could not open video {}'.format(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    span = 1 + (num_frames - 1) * sample_rate
    max_start = max(total_frames - span, 0)
    start_index = max_start // 2
    frames_rgb, sampled_start = input_data.get_video_data(video_path, num_frames, sample_rate, s_index=start_index)
    if not frames_rgb:
        raise RuntimeError('Could not load frames from {}'.format(video_path))

    processed_frames = input_data.data_process(frames_rgb, crop_size)
    frame_array = np.asarray(processed_frames, dtype=np.float32)
    return frame_array, sampled_start


def normalize_cam(cam):
    cam = np.maximum(cam, 0.0)
    if float(cam.max()) <= 0.0:
        return np.zeros_like(cam)
    cam = cam / float(cam.max())
    return cam


def overlay_heatmap(frame_rgb, heatmap, alpha):
    heatmap_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(np.uint8(np.clip(frame_rgb, 0, 255)), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0.0)
    return blended


def make_vertical_separator(height, width=20):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def make_horizontal_separator(width, height=28):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def stack_row(frames_bgr):
    if not frames_bgr:
        raise ValueError('No frames provided for row composition.')
    separator = make_vertical_separator(frames_bgr[0].shape[0])
    row = [frames_bgr[0]]
    for frame in frames_bgr[1:]:
        row.extend([separator, frame])
    return np.hstack(row)


def select_frame_indices(total_frames, display_frames):
    if display_frames >= total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num=display_frames, dtype=int).tolist()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    checkpoint_path = resolve_checkpoint_path(args)
    label_to_class = read_reference_list(args.reference_list)
    clip_array, clip_start = get_center_clip(args.video_path, args.num_frames, args.sample_rate, args.crop_size)
    clip_batch = np.expand_dims(clip_array, axis=0)

    with tf.Graph().as_default():
        rgb_images_placeholder = tf.placeholder(
            tf.float32,
            shape=(1, args.num_frames, args.crop_size, args.crop_size, 3),
            name='rgb_images',
        )
        is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('RGB'):
            rgb_logits, end_points = InceptionI3d(
                num_classes=args.num_classes,
                spatial_squeeze=True,
                final_endpoint='Logits',
            )(rgb_images_placeholder, is_training)

        target_class_tensor = (
            tf.constant(args.target_class, dtype=tf.int32)
            if args.target_class is not None
            else tf.argmax(rgb_logits[0], axis=0, output_type=tf.int32)
        )
        target_score = rgb_logits[0, target_class_tensor]
        mixed_5c = end_points['Mixed_5c']
        grads = tf.gradients(target_score, mixed_5c)[0]
        probabilities = tf.nn.softmax(rgb_logits, axis=1)

        saver = tf.train.Saver()
        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)

            probs_value, target_class, activations_value, grads_value, logits_value = sess.run(
                [probabilities, target_class_tensor, mixed_5c, grads, rgb_logits],
                feed_dict={
                    rgb_images_placeholder: clip_batch,
                    is_training: False,
                },
            )

    activations_value = activations_value[0]
    grads_value = grads_value[0]
    channel_weights = np.mean(grads_value, axis=(0, 1, 2))
    cam = np.tensordot(activations_value, channel_weights, axes=([3], [0]))
    cam = normalize_cam(cam)

    selected_indices = select_frame_indices(cam.shape[0], args.display_frames)
    raw_frames = []
    overlay_frames = []
    for frame_index in selected_indices:
        frame_rgb = clip_array[frame_index]
        resized_cam = cv2.resize(cam[frame_index], (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
        raw_frames.append(cv2.cvtColor(np.uint8(np.clip(frame_rgb, 0, 255)), cv2.COLOR_RGB2BGR))
        overlay_frames.append(overlay_heatmap(frame_rgb, resized_cam, args.overlay_alpha))

    raw_row = stack_row(raw_frames)
    overlay_row = stack_row(overlay_frames)
    composed = np.vstack([raw_row, make_horizontal_separator(raw_row.shape[1]), overlay_row])

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not cv2.imwrite(args.output_path, composed):
        raise RuntimeError('Could not write output image {}'.format(args.output_path))

    top5_indices = np.argsort(probs_value[0])[::-1][:5]
    metadata = {
        'video_path': args.video_path,
        'checkpoint_path': checkpoint_path,
        'clip_start_index': int(clip_start),
        'target_class': int(target_class),
        'target_class_name': label_to_class.get(int(target_class), str(int(target_class))),
        'top5_indices': [int(index) for index in top5_indices],
        'top5_class_names': [label_to_class.get(int(index), str(int(index))) for index in top5_indices],
        'top5_probabilities': [float(probs_value[0][index]) for index in top5_indices],
        'output_path': args.output_path,
        'logits': [float(value) for value in logits_value[0]],
    }
    metadata_path = args.metadata_path or os.path.splitext(args.output_path)[0] + '.json'
    with open(metadata_path, 'w') as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()