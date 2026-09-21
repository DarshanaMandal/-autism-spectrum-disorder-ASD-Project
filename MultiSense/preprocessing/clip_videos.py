"""
MultiSense Preprocessing: SSBD Video Segment Clipper
Extracts annotated behavior intervals (e.g. 18s-24s) from raw videos
and exports precise, behavior-specific clips into data/processed/<Category>/.
"""

import os
import json
import argparse
from pathlib import Path
import cv2


def clip_all_segments(annotations_json_path, raw_dir, processed_dir):
    with open(annotations_json_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    for cat in ["ArmFlapping", "HeadBanging", "Spinning"]:
        os.makedirs(os.path.join(processed_dir, cat), exist_ok=True)

    clipped_records = []
    total_clipped = 0
    skipped_count = 0

    print(f"[*] Starting video clipping from {raw_dir}...")

    for v_idx, video_entry in enumerate(catalog):
        vid_id = video_entry["video_id"]
        primary_cat = video_entry["primary_category"]
        segments = video_entry.get("segments", [])

        # Locate raw video
        raw_video_path = os.path.join(raw_dir, primary_cat, f"{vid_id}.mp4")
        if not os.path.exists(raw_video_path):
            skipped_count += len(segments)
            continue

        cap = cv2.VideoCapture(raw_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur_sec = total_video_frames / fps if fps > 0 else 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 480
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        for seg in segments:
            seg_id = seg["segment_id"]
            cat = seg["category"]
            start_sec = seg["start_sec"]
            end_sec = seg["end_sec"]

            # Clamp timestamps to available video duration
            if video_dur_sec > 0:
                if start_sec >= video_dur_sec:
                    start_sec = max(0.0, video_dur_sec - 3.0)
                    end_sec = video_dur_sec
                elif end_sec > video_dur_sec:
                    end_sec = video_dur_sec

            start_frame = max(0, int(start_sec * fps))
            end_frame = max(start_frame + 15, int(end_sec * fps))
            if end_frame > total_video_frames and total_video_frames > 0:
                end_frame = total_video_frames

            out_clip_path = os.path.join(processed_dir, cat, f"{seg_id}.mp4")

            # Extract frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(out_clip_path, fourcc, fps, (width, height))

            curr_frame = start_frame
            extracted_frames = 0
            while curr_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                extracted_frames += 1
                curr_frame += 1

            writer.release()

            clipped_records.append({
                "segment_id": seg_id,
                "video_id": vid_id,
                "category": cat,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "extracted_frames": extracted_frames,
                "fps": fps,
                "clip_path": out_clip_path
            })
            total_clipped += 1

        cap.release()
        if (v_idx + 1) % 15 == 0:
            print(f"    [{v_idx+1}/{len(catalog)}] Clipped segments from {vid_id}")

    # Save index of clipped segments
    index_path = os.path.join(processed_dir, "clipped_segments_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(clipped_records, f, indent=2)

    print("\n" + "="*60)
    print(f"[✓] Clipping complete!")
    print(f"    - Total segments clipped: {total_clipped}")
    print(f"    - Index saved: {index_path}")
    print("="*60 + "\n")
    return clipped_records


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    annotations_json = base_dir / "data" / "ssbd_annotations.json"
    raw_dir = base_dir / "data" / "raw" / "SSBD"
    processed_dir = base_dir / "data" / "processed"
    clip_all_segments(str(annotations_json), str(raw_dir), str(processed_dir))
