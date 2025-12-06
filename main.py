import cv2

for i in range(10):
    sf_name = f"raw/{i}.png"
    df_name = f"processed/{i}.png"
    color_image = cv2.imread(sf_name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.resize(binary_image, (28, 28))
    cv2.imwrite(df_name, binary_image)
