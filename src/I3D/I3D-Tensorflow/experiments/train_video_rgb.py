import argparse
import os
import sys

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import input_data
from i3d import InceptionI3d


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune RGB I3D directly from video files.')
    parser.add_argument('--train-list', required=True, help='Path to train_rgb.list')
    parser.add_argument('--test-list', required=True, help='Path to test_rgb.list')
    parser.add_argument('--num-classes', required=True, type=int, help='Number of dataset classes')
    parser.add_argument('--pretrained-dir', default=os.path.join(ROOT_DIR, 'checkpoints', 'rgb_imagenet'), help='Pretrained RGB checkpoint directory')
    parser.add_argument('--output-dir', required=True, help='Directory for checkpoints and tensorboard logs')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-frames', type=int, default=32)
    parser.add_argument('--sample-rate', type=int, default=2)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES value')
    return parser.parse_args()


def sparse_ce_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def load_batch(list_path, batch_size, num_frames, sample_rate, crop_size, shuffle):
    rgb_images, _, labels, _, _, _ = input_data.read_clip_and_label(
        filename=list_path,
        batch_size=batch_size,
        num_frames_per_clip=num_frames,
        sample_rate=sample_rate,
        crop_size=crop_size,
        shuffle=shuffle,
        add_flow=False,
    )
    return rgb_images, labels


def read_list_lines(list_path):
    with open(list_path, 'r') as handle:
        return [line for line in handle if line.strip()]


def build_pretrained_rgb_saver():
    variable_map = {}
    for variable in tf.global_variables():
        parts = variable.name.split('/')
        if parts[0] != 'RGB':
            continue
        if 'Adam' in parts[-1]:
            continue
        if len(parts) > 2 and parts[2] == 'Logits':
            continue
        variable_map[variable.name.replace(':0', '')] = variable
    return tf.train.Saver(var_list=variable_map, reshape=True)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    train_log_dir = os.path.join(args.output_dir, 'logs', 'train')
    test_log_dir = os.path.join(args.output_dir, 'logs', 'test')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)

    with tf.Graph().as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        rgb_images_placeholder = tf.placeholder(
            tf.float32,
            shape=(args.batch_size, args.num_frames, args.crop_size, args.crop_size, 3),
            name='rgb_images'
        )
        labels_placeholder = tf.placeholder(tf.int64, shape=(args.batch_size,), name='labels')
        is_training = tf.placeholder(tf.bool, name='is_training')

        learning_rate = tf.train.exponential_decay(
            args.learning_rate,
            global_step,
            decay_steps=max(1, args.max_steps // 3),
            decay_rate=0.1,
            staircase=True,
        )
        optimizer = tf.train.AdamOptimizer(learning_rate)

        with tf.variable_scope('RGB'):
            rgb_logits, _ = InceptionI3d(
                num_classes=args.num_classes,
                spatial_squeeze=True,
                final_endpoint='Logits'
            )(rgb_images_placeholder, is_training)

        loss = sparse_ce_loss(rgb_logits, labels_placeholder)
        acc = accuracy(rgb_logits, labels_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()

        pretrained_saver = build_pretrained_rgb_saver()
        saver = tf.train.Saver(max_to_keep=5)

        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())

            train_list_lines = read_list_lines(args.train_list)
            test_list_lines = read_list_lines(args.test_list)

            if args.pretrained_dir:
                ckpt = tf.train.get_checkpoint_state(args.pretrained_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('Loading pretrained checkpoint {}'.format(ckpt.model_checkpoint_path))
                    pretrained_saver.restore(sess, ckpt.model_checkpoint_path)

            train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            test_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

            for step in range(args.max_steps):
                train_rgb, train_labels = load_batch(
                    train_list_lines,
                    args.batch_size,
                    args.num_frames,
                    args.sample_rate,
                    args.crop_size,
                    shuffle=True,
                )
                sess.run(
                    train_op,
                    feed_dict={
                        rgb_images_placeholder: train_rgb,
                        labels_placeholder: train_labels,
                        is_training: True,
                    },
                )

                if step % args.eval_every == 0 or (step + 1) == args.max_steps:
                    train_summary, train_acc, train_loss = sess.run(
                        [merged, acc, loss],
                        feed_dict={
                            rgb_images_placeholder: train_rgb,
                            labels_placeholder: train_labels,
                            is_training: False,
                        },
                    )
                    train_writer.add_summary(train_summary, step)
                    print('step {} train_acc {:.5f} train_loss {:.5f}'.format(step, train_acc, train_loss))

                    test_rgb, test_labels = load_batch(
                        test_list_lines,
                        args.batch_size,
                        args.num_frames,
                        args.sample_rate,
                        args.crop_size,
                        shuffle=True,
                    )
                    test_summary, test_acc, test_loss = sess.run(
                        [merged, acc, loss],
                        feed_dict={
                            rgb_images_placeholder: test_rgb,
                            labels_placeholder: test_labels,
                            is_training: False,
                        },
                    )
                    test_writer.add_summary(test_summary, step)
                    print('step {} test_acc {:.5f} test_loss {:.5f}'.format(step, test_acc, test_loss))

                if (step + 1) % args.save_every == 0 or (step + 1) == args.max_steps:
                    saver.save(sess, os.path.join(checkpoint_dir, 'i3d_rgb'), global_step=step)


if __name__ == '__main__':
    main()