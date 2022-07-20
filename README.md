# Spherical Image Object Detection
Spherical image object detection is to detect objects in spherical images (360 degree panorams) which have spherical deformations.
<img src="./images/SphObjDet.jpg" alt="Spherical Image Object Detection" />

## Evaluation Metric
As the spherical images have deformations, the traditional axis-aligned rectangles can not be used as the bounding boxes for objects. Here we use spherical rectangles as the bounding boxes for spherical objects.
<p align='center'>
<img src="./images/representation.jpg" style="zoom: 20%;" />
</p>

### Unbiased IoU
Our **Unbiased Spherical IoU** first calculates the area of each spherical rectangle, then calculates the intersection area of the two spherical rectangles. Finally, we compute the spherical IoU.

**First**, the area of each spherical rectangle can be computed according to the following formula. (The derivation is given in the supplementary material of our paper.)

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=A(b_i)%20=%204\arccos(-\sin\frac{\alpha_i}{2}\sin\frac{\beta_i}{2})%20-%202\pi,%20\text{for}%20\%20i%20\in%20\{1,%202\}.">
</p>

**Second**, the computation of intersection area contains 3 steps:

- Step 1. Compute intersection points between boundaries of the two spherical rectangles.

- Step 2. Remove unnecessary points by two sub-steps:
  - Sub-step 1: Remove points outside the two spherical rectangles.
  - Sub-step 2: Remove redundant Points. (This step is not required for most cases.)

- Step 3. Compute all angles and the final intersection area.

**Finally**, the spherical IoU is computed by the following formula

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=IoU(b_1,%20b_2)%20=%20\frac{A(b_1%20\cap%20b_2)}{A(b_1%20\cup%20b_2)}%20=%20\frac{A(b_1%20\cap%20b_2)}{A(b_1)%2BA(b_2)%20-%20A(b_1%20\cap%20b_2)}.">
</p>

### Rotated Bounding FoV (RBFov)
The RBFoV is defined by $(\theta, \phi, \alpha, \beta, \gamma)$, where $\theta$ and $\phi$ are the longitude and latitude coordinates of the object center, and $\alpha$, $\beta$ denote the up-down and left-right field-of-view angles of the object’s occupation, $\gamma$ is represents the angle (clockwise is positive, counterclockwise is negative) of the rotation of the tangent plane of the RBFoV along the axis $\vec{OM}$ (The $M$ is the tangent point $(\theta, \phi)$). The range of values of $\gamma$ is $[-90, 90]$.
<p align="center">
<img src="images/RBFoV.png" style="zoom: 5%;" />
</p>

## PANDORA Dataset
**PANDORA** (PANoramic Detection Dataset for Object with ORientAtion) is a dataset for object detection in panoramic images. It contains 3,000 real-world indoor panoramic images with high-resolution (1920 × 960) and 94,353 bounding boxes of 47 categories that locate from low latitude regions to high latitude regions. Note that the annotations of the bounding box in the PANDORA dataset are defined by **RBFoV** our proposed.
<p align="center">
<img src="images/PANDORA.png" style="zoom: 50%;" />
</p>
The dataset can be downloaded from [here](https://drive.google.com/file/d/1JAGReczN_h3F3mY-mlGTVSeDx-CCJigC/view?usp=sharing). The annotation is in COCO format.

## Spherical Object Detection Methods
We propose two anchor-free object detection methods based on CenterNet. One is called Sph-CenterNet, the other one is called R-CenterNet. These networks take panoramic images as input, and predict heatmaps, offsets, sizes. R-CenterNet additionally predicts rotation angle for each object. 

<p align="center">
<img src="images/RCenterNet.jpg" style="zoom: 8%;" />
</p>
<!-- ## Citing Our Work -->

## Citing our Works

If you use our Unbiased IoU in your research or Sph-CenterNet, please cite our AAAI work as

```BibTeX
@inproceedings{SphIoU,
  title = {Unbiased IoU for Spherical Image Object Detection},
  author = {Dai, Feng and Chen, Bin and Xu, Hang and Ma, Yike and Li, Xiaodong and Feng, Bailan and Yuan, Peng and Yan, Chenggang and Zhao, Qiang},
  booktitle = {Proceedings of AAAI Conference on Artificial Intelligence},
  year = {2022},
  pages = {508-515},
}
```

If you use the PANDORA dataset, RBFoV or R-CenterNet, please cite our ECCV work as 

```BibTeX
@inproceedings{PANDORA,
  title={PANDORA: A Panoramic Detection Dataset for Object with Orientation},
  author={Xu, Hang and Zhao, Qiang and Ma, Yike and Li, Xiaodong and Yuan, Peng and Feng, Bailan and Yan, Chenggang and Dai, Feng},
  booktitle={ECCV},
  year={2022}
}
```
