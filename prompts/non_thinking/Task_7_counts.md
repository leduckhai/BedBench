You are given the block counts of a 3D object, listed by color. The object is composed of blocks with different colors. Your task is to determine the dominant color(s) in the object — the color(s) with the highest block count.

Block counts:

{{counts}}

Only consider colors from the predefined set of possible colors in this world: yellow, cyan, green, magenta, red, orange, orchid, teal, slategray, gold, blue, purple.

Please answer in this format:

```
{
  'answer': list[str],
}
```

Example:

```
{
  'answer': ['red', 'blue'],
}
```
