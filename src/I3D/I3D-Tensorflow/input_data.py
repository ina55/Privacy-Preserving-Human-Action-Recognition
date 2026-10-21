# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for load train data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.webm'}


def _is_video_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def _compute_frame_indices(total_frames, num_frames_per_clip, sample_rate, s_index=-1):
    sample_rate = max(1, int(sample_rate))
    num_frames_per_clip = max(1, int(num_frames_per_clip))
    total_frames = int(total_frames)
    if total_frames <= 0:
        return [], 0

    span = 1 + (num_frames_per_clip - 1) * sample_rate
    if total_frames >= span:
        max_start = total_frames - span
        start_index = random.randint(0, max_start) if s_index < 0 else min(max(int(s_index), 0), max_start)
        frame_indices = [start_index + i * sample_rate for i in range(num_frames_per_clip)]
        return frame_indices, start_index

    if total_frames >= num_frames_per_clip:
        max_start = total_frames - num_frames_per_clip
        start_index = random.randint(0, max_start) if s_index < 0 else min(max(int(s_index), 0), max_start)
        frame_indices = [start_index + i for i in range(num_frames_per_clip)]
        return frame_indices, start_index

    if num_frames_per_clip == 1:
        return [0], 0

    frame_indices = []
    for i in range(num_frames_per_clip):
        position = int(round(i * (total_frames - 1) / float(num_frames_per_clip - 1)))
        frame_indices.append(min(max(position, 0), total_frames - 1))
    return frame_indices, 0


def _read_image_sequence(frame_dir, frame_names, frame_indices):
    ret_arr = []
    for frame_index in frame_indices:
        image_name = os.path.join(frame_dir, frame_names[frame_index])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr


def get_video_data(filename, num_frames_per_clip, sample_rate, s_index=-1):
    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        print('DATA_ERRO: %s' % filename)
        return [], s_index

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        frame_indices, start_index = _compute_frame_indices(len(frames), num_frames_per_clip, sample_rate, s_index)
        if not frame_indices:
            print('DATA_ERRO: %s' % filename)
            return [], s_index
        return [frames[index] for index in frame_indices], start_index

    frame_indices, start_index = _compute_frame_indices(frame_count, num_frames_per_clip, sample_rate, s_index)
    ret_arr = []
    last_valid_frame = None
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            if last_valid_frame is None:
                continue
            frame_rgb = last_valid_frame.copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_valid_frame = frame_rgb
        ret_arr.append(frame_rgb)
    capture.release()

    if not ret_arr:
        print('DATA_ERRO: %s' % filename)
        return [], s_index
    while len(ret_arr) < num_frames_per_clip:
        ret_arr.append(ret_arr[-1].copy())
    return ret_arr, start_index



def sample_data(ori_arr, num_frames_per_clip, sample_rate):
    ret_arr = []
    for i in range(int(num_frames_per_clip/sample_rate)):
        ret_arr.append(ori_arr[int(i*sample_rate)])
    return ret_arr


def get_data(filename, num_frames_per_clip, sample_rate, s_index=-1):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames)==0:
            print('DATA_ERRO: %s'%filename)
            return [], s_index
        filenames = sorted(filenames)
        frame_indices, s_index = _compute_frame_indices(len(filenames), num_frames_per_clip, sample_rate, s_index)
        if not frame_indices:
            return [], s_index
        return _read_image_sequence(filename, filenames, frame_indices), s_index
    return ret_arr, s_index


def get_frames_data(filename, num_frames_per_clip, sample_rate, add_flow):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    if _is_video_file(filename):
        rgb_ret_arr, s_index = get_video_data(filename, num_frames_per_clip, sample_rate)
    else:
        filename_i = os.path.join(filename, 'i')
        rgb_source = filename_i if os.path.isdir(filename_i) else filename
        rgb_ret_arr, s_index = get_data(rgb_source, num_frames_per_clip, sample_rate)
    if not add_flow:
        return rgb_ret_arr, [], s_index
    if _is_video_file(filename):
        print('DATA_ERRO: optical flow loading requires extracted x/y frame folders: %s' % filename)
        return [], [], s_index
    filename_x = os.path.join(filename, 'x')
    flow_x, _ = get_data(filename_x, num_frames_per_clip, sample_rate, s_index)
    flow_x = np.expand_dims(flow_x, axis=-1)
    filename_y = os.path.join(filename, 'y')
    flow_y, _ = get_data(filename_y, num_frames_per_clip, sample_rate, s_index)
    flow_y = np.expand_dims(flow_y, axis=-1)
    flow_ret_arr = np.concatenate((flow_x, flow_y), axis=-1)
    return rgb_ret_arr, flow_ret_arr, s_index


def get_frames(filename, s_index, num_frames_per_clip, crop_size, sample_rate, add_flow, position=-1):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    rgb_ret_arr, s_index = get_data(filename, num_frames_per_clip, sample_rate, int(s_index))
    rgb_ret_arr = data_process_pos(rgb_ret_arr, crop_size, int(position))
    if not add_flow:
        return rgb_ret_arr, [], s_index


def data_process_pos(tmp_data, crop_size, position):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
            if j == 0:
                if position==-1:
                    crop_x = random.randint(0, int(img.shape[0] - crop_size))
                    crop_y = random.randint(0, int(img.shape[1] - crop_size))
                elif position==0:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = 0
                elif position==1:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int((img.shape[1] - crop_size) / 2)
                else:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int(img.shape[1] - crop_size)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
            if j == 0:
                if position==-1:
                    crop_x = random.randint(0, int(img.shape[0] - crop_size))
                    crop_y = random.randint(0, int(img.shape[1] - crop_size))
                elif position==0:
                    crop_x = 0
                    crop_y = int((img.shape[1] - crop_size) / 2)
                elif position==1:
                    crop_x = int((img.shape[0] - crop_size) / 2)
                    crop_y = int((img.shape[1] - crop_size) / 2)
                else:
                    crop_x = int(img.shape[0] - crop_size)
                    crop_y = int((img.shape[1] - crop_size) / 2)

        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)
    return img_datas



def data_process(tmp_data, crop_size):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
        if j == 0:
            # crop_x = random.randint(0, int(img.shape[0] - crop_size))
            # crop_y = random.randint(0, int(img.shape[1] - crop_size))
            crop_x = int((img.shape[0] - crop_size) / 2)
            crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)
    return img_datas

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=64, sample_rate=4, crop_size=224, shuffle=True, add_flow=False):
    if isinstance(filename, (list, tuple)):
        lines = list(filename)
    else:
        with open(filename, 'r') as handle:
            lines = list(handle)
    read_dirnames = []
    rgb_data = []
    flow_data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = list(range(len(lines)))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # Process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if batch_index >= batch_size:
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
        tmp_rgb_data, tmp_flow_data, _ = get_frames_data(dirname, num_frames_per_clip, sample_rate, add_flow)
        if len(tmp_rgb_data) != 0:
            rgb_img_datas = data_process(tmp_rgb_data, crop_size)
            if add_flow:
                flow_img_datas = data_process(tmp_flow_data, crop_size)
                flow_data.append(flow_img_datas)
            rgb_data.append(rgb_img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(rgb_data)
    pad_len = batch_size - valid_len
    if valid_len == 0:
        raise ValueError('No valid clips could be loaded from {}'.format(filename))
    if pad_len:
        for i in range(pad_len):
            rgb_data.append(rgb_data[-1])
            if add_flow:
                flow_data.append(flow_data[-1])
            label.append(int(label[-1]))

    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32) if add_flow else np.array([]).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size), next_batch_start, read_dirnames, valid_len
