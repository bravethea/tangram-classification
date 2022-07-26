import os
from pathlib import Path
from typing import List, TypedDict, Dict

import pympi
import torch
from pose_format import Pose
from torch.utils.data import Dataset

from tqdm import tqdm
from collections import Counter
from shared.pose_utils import pose_normalization_info, pose_hide_legs


class Segment(TypedDict):
    start_time: float
    end_time: float
    gesture: str


class PoseSegmentsDatum(TypedDict):
    id: str
    segments: Dict[str, List[Segment]]
    pose: Pose


CLASSES = {"-": 0, "PG": 1, "IG": 2, "OG": 3, "UG": 4}


def build_classes_vector(timestamps: torch.Tensor, segments: List[Segment]):
    classes = torch.zeros(len(timestamps), dtype=torch.long)

    timestamp_i = 0
    for segment in segments:
        while timestamps[timestamp_i] < segment["start_time"]:
            timestamp_i += 1
        segment_start_i = timestamp_i
        while timestamp_i < len(timestamps) and timestamps[timestamp_i] < segment["end_time"]:
            timestamp_i += 1
        segment_end_i = timestamp_i

        classes[segment_start_i:segment_end_i] = CLASSES[segment["gesture"]]

    return classes


class PoseSegmentsDataset(Dataset):
    def __init__(self, data: List[PoseSegmentsDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def build_classes_vectors(self, datum):
        pose_length = len(datum["pose"].body.data)
        timestamps = torch.div(torch.arange(0, pose_length), datum["pose"].body.fps)

        return {
            "lh": build_classes_vector(timestamps, datum["segments"]["LH"]),
            "rh": build_classes_vector(timestamps, datum["segments"]["RH"])
        }

    def __getitem__(self, index):
        datum: PoseSegmentsDatum = self.data[index]
        pose = datum["pose"]

        torch_body = pose.body.torch()
        pose_data = torch_body.data.tensor[:, 0, :, :]
        return {
            "id": datum["id"],
            **self.build_classes_vectors(datum),
            "mask": torch.ones(len(pose.body.data), dtype=torch.float),
            "pose": {
                "obj": pose,
                "data": pose_data
            }
        }

    def inverse_classes_ratio(self) -> List[float]:
        counter = Counter()
        for datum in self.data:
            classes = self.build_classes_vectors(datum)
            for hand_classes in classes.values():
                counter += Counter(hand_classes.numpy().tolist())
        sum_counter = sum(counter.values())
        print(counter)
        return [sum_counter / counter[i] for c, i in CLASSES.items()]


def elan_file_to_segments(file_name: str) -> Dict[str, List[Segment]]:
    eaf = pympi.Elan.Eaf(file_name)
    segments = {}
    for tier_name in eaf.get_tier_names():
        spkr, hand = tier_name.split("_")
        tier = eaf.tiers[tier_name][0]
        annotations = list(tier.values())
        segments_list = [Segment(start_time=eaf.timeslots[ts_start] / 1000,
                                 end_time=eaf.timeslots[ts_end] / 1000,
                                 gesture=gesture)
                         for (ts_start, ts_end, gesture, _) in annotations]
        segments[hand] = sorted(segments_list, key=lambda s: s["start_time"])
    return segments


def process_datum(pose_file: str, elan_file: str, pose_components: List[str]) -> PoseSegmentsDatum:
    with open(pose_file, "rb") as f:
        pose = Pose.read(f.read())
    # For some reason, data is read only
    # pose.body.data = pose.body.data.copy()
    # pose.body.confidence = pose.body.confidence.copy()
    # Normalize and remove length for consistency
    pose = pose.get_components(pose_components)
    pose.normalize(pose_normalization_info(pose.header))
    pose_hide_legs(pose)

    segments = elan_file_to_segments(elan_file)
    return PoseSegmentsDatum(id=pose_file, pose=pose, segments=segments)


def get_dataset(components: List[str] = None):
    current_directory = Path(__file__).parent

    elan_directory = os.path.join(current_directory, "data", "elan_files")
    pose_directory = os.path.join(current_directory, "data", "pose_files")

    pose_file_names = os.listdir(pose_directory)
    elan_file_names = [f.replace(".mp4", "").replace(".pose", ".eaf") for f in pose_file_names]

    pose_files = [os.path.join(pose_directory, f) for f in pose_file_names]
    elan_files = [os.path.join(elan_directory, f) for f in elan_file_names]

    data = [process_datum(pose_file, elan_file, components)
            for pose_file, elan_file in tqdm(list(zip(pose_files, elan_files)))]
    return PoseSegmentsDataset(data)


if __name__ == "__main__":
    dataset = get_dataset(["POSE_LANDMARKS"])
    print(dataset.inverse_classes_ratio())
