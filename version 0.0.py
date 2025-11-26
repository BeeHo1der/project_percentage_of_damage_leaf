import cv2
import numpy as np

def leaf_disease(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # зелёный в хсв
    lower_green= np.array([25, 40, 40])
    upper_green= np.array([85, 255, 255])

    # маска зеленых областей
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ =cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Не найдено зеленых областей")
        return

    largest_contour = max(contours, key=cv2.contourArea)

    # маска листа
    leaf_mask = np.zeros_like(green_mask)
    cv2.fillPoly(leaf_mask, [largest_contour],255)

    #S листа
    leaf_area = cv2.countNonZero(leaf_mask)

    # здоровая площадь(пересечение зел маски и маски листа)
    healthy_area = cv2.countNonZero(cv2.bitwise_and(leaf_mask, green_mask))

    # процент поражения
    if leaf_area > 0:
        disease_ratio = (leaf_area - healthy_area) / leaf_area
        disease_percentage = disease_ratio * 100
    else:
        disease_percentage = 0

    result = img.copy()
    cv2.drawContours(result, [largest_contour], -1, (255, 0, 0), 2)

    cv2.putText(result, f"Disease: {disease_percentage:.1f}%",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
    cv2.imshow('Original', img)
    cv2.imshow('Green Mask', green_mask)
    cv2.imshow('Leaf Area', leaf_mask)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disease_percentage

disease_percent = leaf_disease('57.png')
print(f"Процент поражения: {disease_percent:.2f}%")