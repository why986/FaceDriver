# Face Super Resolution

This module is based on https://github.com/ewrfcas/Face-Super-Resolution

## Setup

* Download [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1u3h3nX5f_w-HJV8Nd1zwqc3uTnVja5Ol).
* Download pretrained generator weights [90000_G.pth](https://drive.google.com/open?id=1CZkLZPtbJepgksCM93MvsY7NgqnEZSvk).
* Put the above files into `data` folder.

## Example

```python3
import cv2
from super_resolution import SRModel

sr_model = SRModel(gpu_ids='0,1') # assume using gpu 0,1

# Read image from file
# - Remember to convert image format from BGR to RGB!
# - The model accepts numpy.ndarray (RGB format) as input, as well as output.
img = cv2.imread('input.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run the model
sr_model.forward(img)

# Write image to file for preview
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('output.png', img)
```
