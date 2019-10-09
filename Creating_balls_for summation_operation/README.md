## Sum operation on hand written string using balls 

1. Summary - After running the script you have to write a string like 2+3+4, 1+7, 9+3+2+1 on white board. After that click once on the screen then balls will come out 
from every digit and number of balls depend on digit. eg. from 3 - three balls will comes out. 
2. Prerequisite - knowledge of pygame, digit prediction [click for more](https://www.dropbox.com/s/y4hq20s9xhap9hp/handwritten.md?dl=0)

Approach-

1. First we have to detect the string that is written on board. For that we are using a CNN model and then storing the predicted string in variable 's'.
2. In the detection we also detect the x-coordinate of detected number and store in 'X_cor_numb'.
4. Predicted numbers store in a list 'numbs'.
3. Now when we generate new balls via addBall() function we also give coordinate of x_cor_numb as arguement.
4. How many times we call the function addBall() is depends on how many number we want to generate.

## Installation 

you need pyhton3, cv2, numpy.
```bash
pip install opencv-python
pip install opencv-contrib-python
pip install pygame
```

## How to run

```bash
python3 sum_of_balls.py
```

### [Demo video](https://www.dropbox.com/s/3qt08x0uty9ruqh/sum_opern_with_balls.mkv?dl=0)