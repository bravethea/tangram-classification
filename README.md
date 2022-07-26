# tangram

Repository for pose classification for the tangram project.

- [shared](shared) - includes shared utilities for our models
- [pose_to_segments](pose_to_segments) - classifies pose sequences on a frame level

## Data

Download the data from [here](https://drive.google.com/drive/folders/12Ow8S7-wyezrLt2LalittxZj91RTdA4S) and put it in the `pose_to_segments/data` directory.

## Training

```bash
python -m pose_to_segments.train
```