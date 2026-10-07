You are given two perspective images of the same 3D scene from different viewing angles and three candidate 2D orthographic silhouettes of that scene. Each scene contains multiple 3D objects (boxes, cylinders, etc.) arranged together. Only one of the silhouettes belongs to the scene; the other two do not. Your task is to identify the valid silhouette.

Please answer in this format:
```
{
  "candidate_idx": int, // 0, 1, 2
}
```

Example:
```
{
  "candidate_idx": 2,
}
```
