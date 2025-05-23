import cv2
import lYolo as ly

im = cv2.imread("t1.jpg")
obj = ly.LYL()
obj.init(im)

img = obj.drawObj("", True)

#count = obj.countObj("con meo")
#y.settext(img, "so luong : " + str(count) + "", (30,35))
cv2.imshow("return", img)
cv2.waitKey(0)
cv2.destroyAllWindows()