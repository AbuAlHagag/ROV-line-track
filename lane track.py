import cv2
import numpy as np
import functions as fn

path = 'track1.jpg'
rel1 = 50
rel2 = 30
size1 = 1000
size2 = 600
hor = 0
ver = 0
v = 7

last_x = 0
last_y = 0
points = []

while True:
    # cap = cv2.VideoCapture()
    view, line = fn.track(hor, ver, rel1=rel1, resize1=size1)
    contours, mask = fn.getcontours(line)
    if len(contours) > 0:
        box = cv2.minAreaRect(contours[0])
        (cx, cy), (w_box, h_box), ang = box
        box = cv2.boxPoints(box)


        for x_box, y_box in box:
            if (x_box-size2/2)**2+(y_box-size2/2)**2 >= (size2*0.40)**2:
                dis = (abs(int(x_box)-last_x)**2 + abs(int(y_box)-last_y)**2)**0.5
                points.append([int(x_box), int(y_box), int(dis)])

        if len(points) > 2:
                points.sort(key=lambda item: item[2])
                del points[2:4]


        if len(points) == 2:
                last_x = (int(points[0][0])+int(points[1][0]))/2
                last_y = (int(points[0][1])+int(points[1][1]))/2
        if len(points) == 1:
                last_x = points[0][0]
                last_y = points[0][1]
        for x_box, y_box, dis in points:
            cv2.circle(line, (int(x_box), int(y_box)), int(size2*0.02), (255, 0, 0), 3)
        points = []
        box = np.int0(box)
        cv2.drawContours(line, [box], 0, (0, 0, 255), 3)
        cv2.arrowedLine(line, (int(size2 / 2), int(size2 / 2)), (int(last_x), int(last_y)), (255, 0, 255), 7)
        cv2.circle(line, (int(size2 / 2), int(size2 / 2)), int(size2*0.45), (255, 255, 0), 3)
        # cv2.circle(line, (int(last_x), int(last_y)), 3, (0, 255, 255), 10)
        xpoint = last_x - size2/2
        ypoint = size2/2 - last_y
        rpoint = (xpoint**2+ypoint**2)**0.5

        hor = int(hor + v*xpoint/rpoint)
        ver = int(ver - v*ypoint/rpoint)
        #print(int(v*xpoint/rpoint), int(v*ypoint/rpoint))

    cv2.imshow("line", line)
    cv2.imshow("view", view)
    # cv2.imshow("MASK", mask)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
