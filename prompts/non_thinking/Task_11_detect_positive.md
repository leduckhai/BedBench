You are given two perspective images of the same 3D object from different viewing angles and three candidate 2D orthographic projection images. The projections correspond to the three principal planes: XY (top view), YZ (side view), and ZX (front view) in a randomized order. However, only one of the projections is correct and actually belong to the object. Your task is to identify the valid projection.

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