import numpy as np

#还可以使用偏移量，实现多个类别一次nms
def nms(dets, thresh):
    x1 = dets[:, 0] #xmin
    y1 = dets[:, 1] #ymin
    x2 = dets[:, 2] #xmax
    y2 = dets[:, 3] #ymax
    scores = dets[:, 4] #confidence
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort()[::-1] # boundingbox的置信度排序，返回索引下标
    keep = [] # 用来保存最后留下来的boundingbox
    while order.size > 0: #这里记住size不是函数调用，size是属性   
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]]) #交叉区域的左上角的横坐标
        yy1 = np.maximum(y1[i], y1[order[1:]]) #交叉区域的左上角的纵坐标
        xx2 = np.minimum(x2[i], x2[order[1:]]) #交叉区域右下角的横坐标
        yy2 = np.minimum(y2[i], y2[order[1:]]) #交叉区域右下角的纵坐标
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0] #返回的是坐标tuple,[0]表示维度0轴的位置，[1]返回的是1轴的位置，这里好像没有1轴
        order = order[inds + 1] #为什么加1，相当于框0和从1以后的框算的iou，然后返回从0开始的索引
        
    return keep



if __name__=="__main__":
    dets = np.array([
                [204, 102, 358, 250, 0.5],
                [257, 118, 380, 250, 0.7],
                [280, 135, 400, 250, 0.6],
                [255, 118, 360, 235, 0.7],
                [205, 102, 358, 250, 0.5],
                [258, 118, 380, 250, 0.7],
                [281, 135, 400, 250, 0.6],
                [256, 118, 360, 235, 0.7]
                
                ])
    thresh = 0.6
    nms(dets,thresh)
    print(0)