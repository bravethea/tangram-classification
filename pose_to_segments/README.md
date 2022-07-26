# Pose-to-Segments

Pose segmentation & classification model on left and right hand actions independently

## Main Idea

We tag pose sequences with different classes and try to classify each frame. Due to huge sequence sizes intended to
work on (full videos), this is not done using a transformer.

#### Pseudo code:

```python
pose_embedding = embed_pose(pose)
pose_encoding = encoder(pose_embedding)
lh_bio = lh_tagger(pose_encoding)
rh_bio = rh_tagger(pose_encoding)
```

## Extra details

- Model tests, including overfitting, and continuous integration
- We remove the legs because they are not informative
- We remove the face because we only care about hand gestures
- For experiment management we use WANDB
- Training works on CPU and GPU (90% util)
- Multiple-GPUs not tested
