import unittest
from typing import List, Dict

import torch

from ..data import PoseSegmentsDatum, Segment, PoseSegmentsDataset
from shared.pose_utils import fake_pose


def single_datum(num_frames, segments: Dict[str, List[Segment]]) -> PoseSegmentsDatum:
    return {
        "id": "test_id",
        "pose": fake_pose(num_frames=num_frames),
        "segments": segments
    }


class DataTestCase(unittest.TestCase):

    def test_item_without_segments(self):
        datum = single_datum(num_frames=5, segments={"RH": [], "LH": []})
        dataset = PoseSegmentsDataset([datum])
        self.assertEqual(len(dataset), 1)

        pose = dataset[0]["pose"]
        self.assertEqual(pose["data"].shape, (5, 137, 2))

        for hand in ["lh", "rh"]:
            bio = dataset[0][hand]
            self.assertEqual(bio.shape, tuple([5]))
            self.assertTrue(torch.all(torch.eq(torch.zeros_like(bio), bio)))


if __name__ == '__main__':
    unittest.main()
