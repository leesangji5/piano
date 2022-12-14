import cv2

image = cv2.imread("image/pianokey.jpg", cv2.IMREAD_ANYCOLOR)

image = cv2.resize(image, dsize=(650, 500),interpolation=cv2.INTER_AREA )

cv2.imshow("pianokey", image)
cv2.waitKey()
cv2.destroyAllWindows()