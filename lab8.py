import cv2
import numpy as np

template_path = '7.jpg'
fly_path = 'img.png'


template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  #серая
template = cv2.flip(template, -1)  #повернута на 180 градусов
w, h = template.shape[::-1]


fly = cv2.imread(fly_path, cv2.IMREAD_UNCHANGED)
fx, fy = fly.shape[1]//2, fly.shape[0]//2


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Рисуем рамку метки
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)


    center_x = top_left[0] + w // 2
    center_y = top_left[1] + h // 2

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    # Расстояние до центра
    dist_x = center_x - frame_center_x
    dist_y = center_y - frame_center_y
    text = f"Distance to center: X={dist_x}, Y={dist_y}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    # Наложение мухи на центр метки
    cy, cx = center_y - fy, center_x - fx

    # Проверка границ, чтобы не выйти за кадр
    if 0 <= cx < frame.shape[1]-fly.shape[1] and 0 <= cy < frame.shape[0]-fly.shape[0]:
        # Если у мухи есть альфа-канал (прозрачность)
        if fly.shape[2] == 4:
            alpha_s = fly[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[cy:cy+fly.shape[0], cx:cx+fly.shape[1], c] = (
                    alpha_s * fly[:, :, c] + alpha_l * frame[cy:cy+fly.shape[0], cx:cx+fly.shape[1], c]
                )
        else:
            frame[cy:cy+fly.shape[0], cx:cx+fly.shape[1]] = fly

   #show
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()