# lYL
detect object with yolo

![image](https://user-images.githubusercontent.com/44722169/168905929-37573d67-3191-4c73-9fd3-b0732d102d5c.png)

**How to use**

import cv2

import lYolo as ly

im = cv2.imread("t3.jpg")

obj = ly.LYL()

obj.init(im)

#detect and draw rectange all object with label

img = obj.drawObj("", True)

img = obj.drawObj("person", True)
