You are given 5 images in the following order:
- Image 1, 2, 3: Silhouettes
- Image 4: Candidate View 0 (candidate_idx = 0)
- Image 5: Candidate View 1 (candidate_idx = 1)

The three silhouettes all belong to the same 3D object. One of the two candidate views correctly represents that object (positive); the other belongs to a different object (negative). Your task is to identify the positive view.

Please answer in this format:
```
{
  'candidate_idx': int, // 0 or 1
}
```

Example:
```
{
  'candidate_idx': 0,
}
```