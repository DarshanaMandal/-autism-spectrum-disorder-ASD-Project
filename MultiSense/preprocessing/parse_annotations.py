"""
MultiSense Preprocessing: SSBD XML Annotations Parser
Parses all 75 XML annotation files from ssbd-release/Annotations,
standardizes timestamp segments, and exports clean metadata JSON.
"""

import os
import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_duration_to_seconds(dur_str):
    """Convert duration strings like '46s', '1:20', or '80' to float seconds."""
    if not dur_str:
        return 0.0
    dur_str = dur_str.strip().lower().rstrip('s')
    if ':' in dur_str:
        parts = dur_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    try:
        return float(dur_str)
    except ValueError:
        return 0.0


def parse_time_range(time_str):
    """
    Parse time ranges like '18:24' (18s to 24s) or '1:10::1:25' or '10-20'.
    Returns (start_sec, end_sec).
    """
    if not time_str:
        return 0.0, 0.0
    time_str = time_str.strip()
    
    # Check delimiters: '::', ':', '-'
    if '::' in time_str:
        parts = time_str.split('::')
    elif ':' in time_str:
        parts = time_str.split(':')
    elif '-' in time_str:
        parts = time_str.split('-')
    else:
        parts = [time_str, time_str]
        
    if len(parts) == 2:
        start_sec = parse_duration_to_seconds(parts[0])
        end_sec = parse_duration_to_seconds(parts[1])
        if end_sec < start_sec:
            # Swap if inverted
            start_sec, end_sec = end_sec, start_sec
        return start_sec, end_sec
    return 0.0, 0.0


def normalize_category_name(cat_str):
    """Normalize behavior class name into standard CamelCase."""
    cat = cat_str.strip().lower()
    if 'arm' in cat or 'flap' in cat:
        return 'ArmFlapping'
    elif 'head' in cat or 'bang' in cat:
        return 'HeadBanging'
    elif 'spin' in cat:
        return 'Spinning'
    return cat.capitalize()


def parse_all_annotations(annotations_dir, output_json_path):
    xml_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    print(f"[+] Found {len(xml_files)} annotation files in {annotations_dir}")

    catalog = []
    category_summary = {"ArmFlapping": 0, "HeadBanging": 0, "Spinning": 0}
    total_segments = 0

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"[!] Error parsing {xml_path}: {e}")
            continue

        video_id = root.attrib.get("id", Path(xml_path).stem)
        # Determine primary category from filename/id
        if "armflapping" in video_id.lower():
            primary_cat = "ArmFlapping"
        elif "headbanging" in video_id.lower():
            primary_cat = "HeadBanging"
        elif "spinning" in video_id.lower():
            primary_cat = "Spinning"
        else:
            primary_cat = "Unknown"

        url_elem = root.find("url")
        url = url_elem.text.strip() if (url_elem is not None and url_elem.text) else ""
        
        dur_elem = root.find("duration")
        dur_raw = dur_elem.text.strip() if (dur_elem is not None and dur_elem.text) else ""
        duration_sec = parse_duration_to_seconds(dur_raw)

        height = int(root.find("height").text) if root.find("height") is not None and root.find("height").text else 0
        width = int(root.find("width").text) if root.find("width") is not None and root.find("width").text else 0
        frames = int(root.find("frames").text) if root.find("frames") is not None and root.find("frames").text else 0
        persons = int(root.find("persons").text) if root.find("persons") is not None and root.find("persons").text else 1

        behaviours_elem = root.find("behaviours")
        segment_list = []

        if behaviours_elem is not None:
            for b_idx, b in enumerate(behaviours_elem.findall("behaviour")):
                cat_elem = b.find("category")
                raw_cat = cat_elem.text.strip() if (cat_elem is not None and cat_elem.text) else primary_cat
                cat_normalized = normalize_category_name(raw_cat)

                time_elem = b.find("time")
                time_str = time_elem.text.strip() if (time_elem is not None and time_elem.text) else ""
                start_sec, end_sec = parse_time_range(time_str)

                # If end_sec <= start_sec or 0, provide default duration fallback
                if end_sec <= start_sec:
                    end_sec = start_sec + 5.0

                intensity_elem = b.find("intensity")
                intensity = intensity_elem.text.strip() if (intensity_elem is not None and intensity_elem.text) else "medium"

                bodypart_elem = b.find("bodypart")
                bodypart = bodypart_elem.text.strip() if (bodypart_elem is not None and bodypart_elem.text) else "body"

                b_id = b.attrib.get("id", f"b_{b_idx+1:02d}")

                segment = {
                    "segment_id": f"{video_id}_{b_id}",
                    "category": cat_normalized,
                    "time_str": time_str,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": round(end_sec - start_sec, 2),
                    "intensity": intensity,
                    "bodypart": bodypart
                }
                segment_list.append(segment)
                if cat_normalized in category_summary:
                    category_summary[cat_normalized] += 1
                total_segments += 1

        video_record = {
            "video_id": video_id,
            "primary_category": primary_cat,
            "url": url,
            "duration_sec": duration_sec,
            "height": height,
            "width": width,
            "frames": frames,
            "persons": persons,
            "segments": segment_list
        }
        catalog.append(video_record)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)

    print(f"[+] Successfully saved {len(catalog)} video records to {output_json_path}")
    print(f"[+] Total behavior segments extracted: {total_segments}")
    for cat, cnt in category_summary.items():
        print(f"    - {cat}: {cnt} segments")

    return catalog


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ann_path = base_dir.parent / "ssbd-release" / "Annotations"
    out_json = base_dir / "data" / "ssbd_annotations.json"
    parse_all_annotations(str(ann_path), str(out_json))
