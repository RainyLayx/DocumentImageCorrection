import numpy as np
import cv2
def extend(x1,y1,x2,y2,w,h):
    if x1==x2:
        x1n = x1
        y1n = 0
        x2n = x1
        y2n = h
    elif y1==y2:
        x1n = 0
        y1n = y1
        x2n = w
        y2n = y1
    else:
        k = (y2 - y1) / (x2 - x1)
        b = (x1 * y2 - x2 * y1) / (x1 - x2)
        xj = round(-b/k)
        yj = round(b)
        if xj>=0 and xj<w and yj>=0 and yj<h:#
            x1n = xj
            y1n = 0
            x2n = 0
            y2n = yj
        elif xj>=0 and xj<w and round(k*w+b)>=0 and round(k*w+b)<h:
            x1n = xj
            y1n = 0
            x2n = w
            y2n = round(k*w+b)
        elif xj>=0 and xj<w and round((h-b)/k)>=0 and round((h-b)/k)<w:
            x1n = xj
            y1n = 0
            x2n = round((h-b)/k)
            y2n = h
        elif round((h-b)/k)>=0 and round((h-b)/k)<w and yj>=0 and yj<h:
            x1n = round((h-b)/k)
            y1n = h
            x2n = 0
            y2n = yj
        elif round((h-b)/k)>=0 and round((h-b)/k)<w and round(k*w+b)>=0 and round(k*w+b)<h:
            x1n = round((h-b)/k)
            y1n = h
            x2n = w
            y2n = round(k*w+b)
        elif yj>=0 and yj<h and round(k*w+b)>=0 and round(k*w+b)<h:
            x1n = 0
            y1n = yj
            x2n = w
            y2n = round(k*w+b)
#         else:
#             return 0,0,0,0
    return x1n,y1n,x2n,y2n


def getFootPoint(x0, y0, x1, y1, x2, y2):
    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1.0

    xf = k * (x2 - x1) + x1
    yf = k * (y2 - y1) + y1

    return xf, yf


def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def polygon_area(polygon):
    """
    计算多边形的面积，支持非凸情况
    :param polygon: 多边形顶点，已经进行顺次逆时针排序
    :return: 该多边形的面积
    """
    n = len(polygon)

    if n < 3:
        return 0

    vectors = np.zeros((n, 2))
    for i in range(0, n):
        vectors[i, :] = polygon[i, :] - polygon[0, :]

    area = 0
    for i in range(1, n):
        area = area + coss_multi(vectors[i - 1, :], vectors[i, :]) / 2

    return area


def four_point_transform(image, tl, tr, br, bl):
    # 获取输入坐标点 左上右上右下左下
    rect = np.array([tl, tr, br, bl], dtype="float32")
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped


def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]

def extend(x1,y1,x2,y2,w,h):
    if x1==x2:
        x1n = x1
        y1n = 0
        x2n = x1
        y2n = h
    elif y1==y2:
        x1n = 0
        y1n = y1
        x2n = w
        y2n = y1
    else:
        k = (y2 - y1) / (x2 - x1)
        # b = (x1 * y2 - x2 * y1) / (x1 - x2)
        b = y1-round(k,2)*x1
        xj = round(-b/k)
        yj = round(b)
        # print(k,b,xj,round(k*w+b),round((h-b)/k),yj)
        if xj>=0 and xj<w and yj>=0 and yj<=h:#
            x1n = xj
            y1n = 0
            x2n = 0
            y2n = yj
        elif xj>=0 and xj<w and round(k*w+b)>=0 and round(k*w+b)<=h:
            x1n = xj
            y1n = 0
            x2n = w
            y2n = round(k*w+b)
        elif xj>=0 and xj<w and round((h-b)/k)>=0 and round((h-b)/k)<=w:
            x1n = xj
            y1n = 0
            x2n = round((h-b)/k)
            y2n = h
        elif round((h-b)/k)>=0 and round((h-b)/k)<w and yj>=0 and yj<=h:
            x1n = round((h-b)/k)
            y1n = h
            x2n = 0
            y2n = yj
        elif round((h-b)/k)>=0 and round((h-b)/k)<w and round(k*w+b)>=0 and round(k*w+b)<=h:
            x1n = round((h-b)/k)
            y1n = h
            x2n = w
            y2n = round(k*w+b)
        elif yj>=0 and yj<h and round(k*w+b)>=0 and round(k*w+b)<=h:
            x1n = 0
            y1n = yj
            x2n = w
            y2n = round(k*w+b)
#         else:
#             return 0,0,0,0
    return x1n,y1n,x2n,y2n


# 逆时针排序四个点
def order_points(pts):
    ''' sort rectangle points by clockwise '''
    sort_x = pts[np.argsort(pts[:, 0]), :]  # np.argsort从小到大排序，返回index  再以第一列为准按index排序
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:, 1]), :]  #::-1 一维数组的逆序排列
    tl = list(Left[0])
    bl = list(Left[1])
    # Right sort
    Right = Right[np.argsort(Right[:, 1]), :]
    tr = list(Right[0])
    br = list(Right[1])

    res = [tl, tr, br, bl]
    return res