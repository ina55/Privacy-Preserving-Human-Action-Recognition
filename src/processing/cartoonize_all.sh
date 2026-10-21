#!/bin/bash

# Usage: ./cartoonize_all.sh
# Cartoonizes all .avi videos in hmdb51_1, preserving action_class structure

SRC_ROOT="/path/to/project/datasets/hmdb51/hmdb51_1"
OUT_ROOT="/path/to/project/datasets/hmdb51/hmdb51_cartoon"
VIDEO_INPUT="/tmp/VideoToCartoon/video_input"
VIDEO_OUTPUT="/tmp/VideoToCartoon/video_output"
CARTOON_ROOT="/tmp/VideoToCartoon"


find "$SRC_ROOT" -type f -name '*.avi' | while read -r src_path; do
    rel_path="${src_path#$SRC_ROOT/}"
    action_class="${rel_path%%/*}"
    video_name="${rel_path##*/}"
    video_name_noext="${video_name%.avi}"
    out_dir="$OUT_ROOT/$action_class"
    out_path="$out_dir/$video_name"

    # Skip if cartoonized video already exists
    if [ -f "$out_path" ]; then
        echo "Skipping $src_path, cartoonized video already exists."
        continue
    fi

    echo "Processing $src_path -> $out_path"

    # Clean input/output folders
    rm -f "$VIDEO_INPUT"/*
    rm -f "$VIDEO_OUTPUT"/*

    # Copy input video
    cp "$src_path" "$VIDEO_INPUT/$video_name"

    # Run VideoToCartoon
    cd "$CARTOON_ROOT"
    python3 main.py --input "$video_name"

    # Move cartoonized output
    mkdir -p "$out_dir"
    mv "$VIDEO_OUTPUT/${video_name_noext}.avi" "$out_path"
done

echo "Batch cartoonization complete."
