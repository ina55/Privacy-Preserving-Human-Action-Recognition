import argparse
import json
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import input_data
from i3d import InceptionI3d


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RGB I3D checkpoints on a video list.')
    parser.add_argument('--test-list', required=True, help='Path to test_rgb.list')
    parser.add_argument('--num-classes', required=True, type=int, help='Number of dataset classes')
    parser.add_argument('--checkpoint-dir', help='Directory containing TensorFlow checkpoint files')
    parser.add_argument('--checkpoint-path', help='Explicit checkpoint prefix, e.g. /path/to/i3d_rgb-2499')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--sample-rate', type=int, default=2)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for quick smoke tests')
    parser.add_argument('--start-pos', type=int, default=0, help='Zero-based starting sample index for sequential resume runs')
    parser.add_argument('--log-every', type=int, default=25, help='Progress log frequency in batches')
    parser.add_argument('--results-dir', help='Directory where evaluation outputs will be written')
    parser.add_argument('--report-path', help='Optional explicit text report path')
    parser.add_argument('--raw-json-path', help='Optional explicit raw JSON path')
    parser.add_argument('--partial-json-path', help='Optional path for partial progress checkpoints')
    parser.add_argument('--save-partial-every', type=int, default=0, help='Save partial JSON every N batches; disabled when 0')
    return parser.parse_args()


def sparse_ce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def resolve_checkpoint_path(args):
    if args.checkpoint_path:
        return args.checkpoint_path
    if not args.checkpoint_dir:
        raise ValueError('Either --checkpoint-path or --checkpoint-dir must be provided.')
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    raise ValueError('No checkpoint found in {}'.format(args.checkpoint_dir))


def count_lines(path):
    with open(path, 'r') as handle:
        return sum(1 for line in handle if line.strip())


def read_list_entries(path):
    entries = []
    with open(path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            video_path, label = line.rsplit(' ', 1)
            entries.append((video_path, int(label)))
    return entries


def build_label_map(entries):
    label_to_class = {}
    for video_path, label in entries:
        class_name = os.path.basename(os.path.dirname(video_path))
        label_to_class.setdefault(label, class_name)
    return label_to_class


def infer_results_dir_from_list(test_list):
    metadata_path = os.path.join(os.path.dirname(test_list), 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as handle:
            metadata = json.load(handle)
        dataset_root = metadata.get('dataset_root')
        if dataset_root:
            dataset_parent = os.path.dirname(dataset_root)
            family_name = os.path.basename(dataset_parent)
            return os.path.join(dataset_parent, family_name + '_results')
    return os.path.join(os.path.dirname(test_list), 'eval_results')


def resolve_output_paths(args, checkpoint_path):
    results_dir = args.results_dir or infer_results_dir_from_list(args.test_list)
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_name = os.path.basename(checkpoint_path)
    report_path = args.report_path or os.path.join(results_dir, 'eval_report_{}.txt'.format(checkpoint_name))
    raw_json_path = args.raw_json_path or os.path.join(results_dir, 'eval_results_{}.json'.format(checkpoint_name))
    return results_dir, report_path, raw_json_path


def compute_topk(probabilities, k):
    indices = np.argsort(probabilities)[::-1][:k]
    return [(int(index), float(probabilities[index])) for index in indices]


def write_report(report_path, summary, per_class_rows, checkpoint_path, test_list, processed):
    lines = []
    lines.append('=' * 80)
    lines.append('HMDB51 EVALUATION REPORT (I3D RGB CHECKPOINT)')
    lines.append('=' * 80)
    lines.append('')
    lines.append('Checkpoint: {}'.format(checkpoint_path))
    lines.append('Test list: {}'.format(test_list))
    lines.append('Total Videos Evaluated: {}'.format(processed))
    lines.append('')
    lines.append('--- OVERALL SUMMARY ---')
    lines.append('Top-1 Accuracy: {:.2f}% ({}/{})'.format(summary['top1_acc'] * 100.0, summary['top1_correct'], processed))
    lines.append('Average Loss: {:.5f}'.format(summary['avg_loss']))
    lines.append('')
    lines.append('--- PER-CLASS BREAKDOWN ---')
    lines.append('Class Name           | Top-1 Acc | Correct | Total')
    lines.append('-' * 60)
    for row in per_class_rows:
        lines.append('{:<20} | {:>8.2f}% | {:>7} | {:>5}'.format(
            row['class_name'],
            row['top1_acc'] * 100.0,
            row['correct'],
            row['total'],
        ))
    with open(report_path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def dump_results(json_path, summary, per_class_rows, raw_results):
    with open(json_path, 'w') as handle:
        json.dump({
            'summary': summary,
            'per_class': per_class_rows,
            'results': raw_results,
        }, handle, indent=2)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    checkpoint_path = resolve_checkpoint_path(args)
    results_dir, report_path, raw_json_path = resolve_output_paths(args, checkpoint_path)
    total_samples = count_lines(args.test_list)
    if args.start_pos < 0 or args.start_pos > total_samples:
        raise ValueError('start_pos must be between 0 and {}'.format(total_samples))
    remaining_samples = total_samples - args.start_pos
    requested_samples = min(remaining_samples, args.max_samples) if args.max_samples else remaining_samples
    test_entries = read_list_entries(args.test_list)
    label_to_class = build_label_map(test_entries)

    with tf.Graph().as_default():
        rgb_images_placeholder = tf.placeholder(
            tf.float32,
            shape=(args.batch_size, args.num_frames, args.crop_size, args.crop_size, 3),
            name='rgb_images'
        )
        labels_placeholder = tf.placeholder(tf.int64, shape=(args.batch_size,), name='labels')
        is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope('RGB'):
            rgb_logits, _ = InceptionI3d(
                num_classes=args.num_classes,
                spatial_squeeze=True,
                final_endpoint='Logits'
            )(rgb_images_placeholder, is_training)

        loss = sparse_ce_loss(rgb_logits, labels_placeholder)
        acc = accuracy(rgb_logits, labels_placeholder)
        predictions = tf.argmax(rgb_logits, axis=1, output_type=tf.int64)

        saver = tf.train.Saver()

        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            print('Loading evaluation checkpoint {}'.format(checkpoint_path))
            saver.restore(sess, checkpoint_path)

            start_pos = args.start_pos
            processed = 0
            batch_count = 0
            total_correct = 0
            total_loss = 0.0
            raw_results = []
            per_class_counts = {
                class_name: {'correct': 0, 'total': 0}
                for _, class_name in sorted(label_to_class.items())
            }

            while start_pos >= 0 and processed < requested_samples:
                rgb_batch, _, labels_batch, next_start, read_dirnames, valid_len = input_data.read_clip_and_label(
                    filename=args.test_list,
                    batch_size=args.batch_size,
                    start_pos=start_pos,
                    num_frames_per_clip=args.num_frames,
                    sample_rate=args.sample_rate,
                    crop_size=args.crop_size,
                    shuffle=False,
                    add_flow=False,
                )

                if processed + valid_len > requested_samples:
                    valid_len = requested_samples - processed

                batch_predictions, batch_logits, batch_loss, batch_acc = sess.run(
                    [predictions, rgb_logits, loss, acc],
                    feed_dict={
                        rgb_images_placeholder: rgb_batch,
                        labels_placeholder: labels_batch,
                        is_training: False,
                    },
                )

                valid_predictions = batch_predictions[:valid_len]
                valid_labels = labels_batch[:valid_len]
                valid_logits = batch_logits[:valid_len]
                total_correct += int(np.sum(valid_predictions == valid_labels))
                total_loss += float(batch_loss) * valid_len

                for item_index in range(valid_len):
                    label = int(valid_labels[item_index])
                    predicted = int(valid_predictions[item_index])
                    class_name = label_to_class.get(label, str(label))
                    per_class_counts.setdefault(class_name, {'correct': 0, 'total': 0})
                    per_class_counts[class_name]['total'] += 1
                    if predicted == label:
                        per_class_counts[class_name]['correct'] += 1

                    probabilities = tf.nn.softmax(valid_logits[item_index]).eval(session=sess)
                    top5 = compute_topk(probabilities, 5)
                    raw_results.append({
                        'video_path': read_dirnames[item_index],
                        'label': label,
                        'class_name': class_name,
                        'top1_pred': predicted,
                        'top1_pred_class': label_to_class.get(predicted, str(predicted)),
                        'top5_preds': [index for index, _ in top5],
                        'top5_pred_classes': [label_to_class.get(index, str(index)) for index, _ in top5],
                        'probs': [score for _, score in top5],
                        'correct_top1': bool(predicted == label),
                    })

                processed += valid_len
                batch_count += 1

                if args.partial_json_path and args.save_partial_every > 0 and batch_count % args.save_partial_every == 0:
                    partial_summary = {
                        'checkpoint_path': checkpoint_path,
                        'test_list': args.test_list,
                        'start_pos': args.start_pos,
                        'samples_evaluated': processed,
                        'absolute_processed': args.start_pos + processed,
                        'top1_correct': total_correct,
                        'top1_acc': (total_correct / float(processed)) if processed else 0.0,
                        'avg_loss': (total_loss / float(processed)) if processed else 0.0,
                        'results_dir': results_dir,
                        'report_path': report_path,
                        'raw_json_path': raw_json_path,
                    }
                    partial_rows = []
                    for class_name in sorted(per_class_counts):
                        total = per_class_counts[class_name]['total']
                        correct = per_class_counts[class_name]['correct']
                        partial_rows.append({
                            'class_name': class_name,
                            'correct': correct,
                            'total': total,
                            'top1_acc': (correct / float(total)) if total else 0.0,
                        })
                    dump_results(args.partial_json_path, partial_summary, partial_rows, raw_results)

                if batch_count % args.log_every == 0 or processed == requested_samples:
                    running_acc = total_correct / float(processed)
                    running_loss = total_loss / float(processed)
                    print(
                        'processed {} / {} videos | running_acc {:.5f} | running_loss {:.5f} | last_batch_acc {:.5f}'.format(
                            args.start_pos + processed,
                            total_samples,
                            running_acc,
                            running_loss,
                            float(batch_acc),
                        )
                    )

                if next_start < 0 or processed >= requested_samples:
                    break
                start_pos = next_start

            final_acc = total_correct / float(processed)
            final_loss = total_loss / float(processed)
            per_class_rows = []
            for class_name in sorted(per_class_counts):
                total = per_class_counts[class_name]['total']
                correct = per_class_counts[class_name]['correct']
                per_class_rows.append({
                    'class_name': class_name,
                    'correct': correct,
                    'total': total,
                    'top1_acc': (correct / float(total)) if total else 0.0,
                })

            summary = {
                'checkpoint_path': checkpoint_path,
                'test_list': args.test_list,
                'start_pos': args.start_pos,
                'samples_evaluated': processed,
                'absolute_processed': args.start_pos + processed,
                'top1_correct': total_correct,
                'top1_acc': final_acc,
                'avg_loss': final_loss,
                'results_dir': results_dir,
                'report_path': report_path,
                'raw_json_path': raw_json_path,
            }

            dump_results(raw_json_path, summary, per_class_rows, raw_results)

            write_report(report_path, summary, per_class_rows, checkpoint_path, args.test_list, processed)
            print('evaluation_complete samples={} top1_acc={:.5f} avg_loss={:.5f}'.format(processed, final_acc, final_loss))
            print('saved_report {}'.format(report_path))
            print('saved_raw_json {}'.format(raw_json_path))


if __name__ == '__main__':
    main()