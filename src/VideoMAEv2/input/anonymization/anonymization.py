import os
import subprocess
from pathlib import Path
from blur_faces import blur_faces_in_video
from swap_faces import swap_faces_in_video


def anonymize_video(input_path,
                    output_dir="./data/VideoMAEv2_data/k400",
                    mode=None):
    input_path = str(input_path)
    os.makedirs(output_dir, exist_ok=True)
    filename = Path(input_path).stem
    output_path = os.path.join(output_dir, f"anonymized_{filename}.mp4")

    selected_mode = (mode or os.getenv("ANON_MODE", "blur")).strip().lower()
    if selected_mode not in {"blur", "swap", "animate"}:
        print(f"Unknown ANON_MODE={selected_mode}; falling back to blur")
        selected_mode = "blur"

    py39 = Path("/usr/bin/python3.9")
    helper_script = None
    if selected_mode in {"swap", "blur"}:
        script_name = "swap_faces.py" if selected_mode == "swap" else "blur_faces.py"
        helper_script = Path(__file__).with_name(script_name)
    else:
        helper_script = Path(__file__).resolve().parents[2] / "scripts" / "motion_transfer" / "pose_animation_pipeline.py"

    if selected_mode == "animate":
        cmd = [
            "python3",
            str(helper_script),
            "--input-video",
            input_path,
            "--output-video",
            output_path,
        ]
        mimic_root = os.getenv("ANIM_MIMIC_ROOT")
        sample_stride = os.getenv("ANIM_SAMPLE_STRIDE", "1")
        if mimic_root:
            cmd.extend(["--mimic-root", mimic_root])
        cmd.extend(["--sample-stride", sample_stride])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            if result.stdout:
                print(result.stdout)
            print(f"✓ Anonymized video saved ({selected_mode}): {output_path}")
            return output_path
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("Animate mode failed.")
        return None

    if py39.exists():
        cmd = [str(py39), str(helper_script), input_path, output_path]
        if selected_mode == "swap":
            reference_dir = os.getenv("ANON_REF_DIR")
            donor_faces = os.getenv("ANON_DONOR_FACES", "5")
            allow_self = os.getenv("ANON_ALLOW_SELF_DONOR", "0").strip() in {"1", "true", "yes"}
            style_randomize = os.getenv("ANON_STYLE_RANDOMIZE", "0").strip() in {"1", "true", "yes"}
            age_mode = os.getenv("ANON_AGE_MODE", "auto").strip().lower()
            if reference_dir:
                cmd.extend(["--reference-dir", reference_dir])
            cmd.extend(["--donor-faces", donor_faces])
            if allow_self:
                cmd.append("--allow-self-donor")
            if style_randomize:
                cmd.append("--style-randomize")
            if age_mode in {"auto", "none", "younger", "older"}:
                cmd.extend(["--age-mode", age_mode])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            if result.stdout:
                print(result.stdout)
            print(f"✓ Anonymized video saved ({selected_mode}): {output_path}")
            return output_path

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Python 3.9 {selected_mode} run failed; falling back to local runtime.")

    try:
        if selected_mode == "swap":
            swap_faces_in_video(
                input_path,
                output_path,
                reference_dir=os.getenv("ANON_REF_DIR"),
                donor_faces=max(2, int(os.getenv("ANON_DONOR_FACES", "5"))),
                allow_self_donor=os.getenv("ANON_ALLOW_SELF_DONOR", "0").strip() in {"1", "true", "yes"},
                style_randomize=os.getenv("ANON_STYLE_RANDOMIZE", "0").strip() in {"1", "true", "yes"},
                age_mode=os.getenv("ANON_AGE_MODE", "auto").strip().lower(),
            )
        else:
            blur_faces_in_video(input_path, output_path)
    except Exception as exc:
        print(f"In-process {selected_mode} failed ({exc}); retrying via Python 3.9 runtime...")
        cmd = [str(py39), str(helper_script), input_path, output_path]
        if selected_mode == "swap":
            reference_dir = os.getenv("ANON_REF_DIR")
            donor_faces = os.getenv("ANON_DONOR_FACES", "5")
            allow_self = os.getenv("ANON_ALLOW_SELF_DONOR", "0").strip() in {"1", "true", "yes"}
            style_randomize = os.getenv("ANON_STYLE_RANDOMIZE", "0").strip() in {"1", "true", "yes"}
            age_mode = os.getenv("ANON_AGE_MODE", "auto").strip().lower()
            if reference_dir:
                cmd.extend(["--reference-dir", reference_dir])
            cmd.extend(["--donor-faces", donor_faces])
            if allow_self:
                cmd.append("--allow-self-donor")
            if style_randomize:
                cmd.append("--style-randomize")
            if age_mode in {"auto", "none", "younger", "older"}:
                cmd.extend(["--age-mode", age_mode])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)

    if os.path.exists(output_path):
        print(f"✓ Anonymized video saved ({selected_mode}): {output_path}")
        return output_path
    print(f"✗ Failed to anonymize video: {input_path}")
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/VideoMAEv2_data/k400"
        mode = sys.argv[3] if len(sys.argv) > 3 else None
        anonymize_video(input_video, output_dir, mode=mode)
    else:
        print("Usage: python anonymization.py <input_video> [output_dir] [mode]")
        sys.exit(1)