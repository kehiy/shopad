import cv2

for i in range(10):
    f_name = f"examples/{i}.png"
    color_image = cv2.imread(f_name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f_name, binary_image)
