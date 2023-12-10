import cv2
import numpy as np
path = 'track1.jpg'


def getcontours(img, edge=False, cThr=[50, 50],finalconts=False, corners=0, minArea=0):
    mask = cv2.GaussianBlur(img, (5, 5), 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if edge:
        mask = cv2.Canny(mask, cThr[0], cThr[1])
    else:
        mask = cv2.bitwise_not(mask)
        thresh, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5))
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.erode(mask, kernel, iterations=3)
    conts, hiearchy =cv2.findContours(mask, -1, cv2.CHAIN_APPROX_SIMPLE)

    if finalconts:
        finalcontours = []
        for i in conts:
            area = cv2.contourArea(i)
            if area > minArea:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02*peri, True)
                bbox = cv2.boundingRect(approx)

                if corners > 0:
                    if len(approx) == corners:
                        finalcontours.append([len(approx), area, approx, bbox, i])
                else:
                    finalcontours.append([len(approx), area, approx, bbox, i])
        finalcontours.sort(key= lambda x:x[1], reverse=True)
        for con in finalcontours:
            cv2.drawContours(img, con[4], -1, (0, 255, 0), 3)
        conts = finalcontours
    else:
        cv2.drawContours(img, conts, -1, (0, 255, 0), 2)

    return conts, mask



def track(x=0, y=0, rel1=50, rel2=40, resize1=1000, resize2=600 ):
    view = cv2.imread(path, 1)
    roi = cv2.bilateralFilter(view, 5, 75, 75)
    roi = cv2.GaussianBlur(roi, (21, 21), 0)

    test = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    start = cv2.HoughCircles(test, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=50, param2=30, minRadius=0, maxRadius=20)
    start = np.uint16(np.around(start))
    xc = start[0, 0, 0]
    yc = start[0, 0, 1]
    view = view[yc - rel1 + y: yc + rel1 + y, xc - rel1 + x: xc + rel1 + x]
    roi = roi[yc - rel2 + y: yc + rel2 + y, xc - rel2 + x: xc + rel2 + x]
    view = cv2.resize(view, (resize1, resize1))
    roi = cv2.resize(roi, (resize2, resize2))
    return view, roi
