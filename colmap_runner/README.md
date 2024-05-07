## COLMAP RUNNER

### Install

```
pip install colmap
```

### Run

```
python img2poses.py
```

#### Explained `poses_bounds.py` file formart

This file stored a numpy array of size Nx17(where N is the number of input images). Each row of length 17 gets reshaped into a 3x5 pose matrix and 2 depth values that bound the closest and farthest scene content from that point of view.

The pose matrix is a 3x4 camera-to-world affine transform concatenated with a 3x1 column `[image height,image width, focal length]` to represent the intrinsics (we assume the principal point is centered and that the focal length is the same for both x and y)

The right-handed coordinate system of the rotation (first 3x3 block in the camera-to-world transform) is as follow: from the point of view the camera, the three axes are `[down,right,backwards]` which some people might consider to be `[-y,x,z]`, where the camera is looking along `-z`. (The more conventional frame `[x,y,z]` is `[right,up,back]`. The COLMAP frame is `[right,down,forwards]` or `[x,y,-z]` )
