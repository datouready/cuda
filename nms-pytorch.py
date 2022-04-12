import torch

def nms(dets, thresh):
    x1 = dets[:, 0] #xmin
    y1 = dets[:, 1] #ymin
    x2 = dets[:, 2] #xmax
    y2 = dets[:, 3] #ymax
    scores = dets[:, 4] #confidence
    print(x1.dtype)
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 每个boundingbox的面积
    order = scores.argsort(0,True) # boundingbox的置信度排序
    # order=torch.tensor(order)
    keep = [] # 用来保存最后留下来的boundingbox
    while len(order) > 0:     
        i = order[0] # 置信度最高的boundingbox的index
        keep.append(i) # 添加本次置信度最高的boundingbox的index
        
        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = torch.maximum(x1[i], x1[order[1:]]) #交叉区域的左上角的横坐标
        yy1 = torch.maximum(y1[i], y1[order[1:]]) #交叉区域的左上角的纵坐标
        xx2 = torch.minimum(x2[i], x2[order[1:]]) #交叉区域右下角的横坐标
        yy2 = torch.minimum(y2[i], y2[order[1:]]) #交叉区域右下角的纵坐标
        
        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = torch.maximum(torch.tensor([0.0]), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor([0.0]), yy2 - yy1 + 1)
        inter = w * h
        
        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留交集小于一定阈值的boundingbox
        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep

if __name__=="__main__":
    dets = torch.tensor([
                [204, 102, 358, 250, 0.5],
                [257, 118, 380, 250, 0.7],
                [280, 135, 400, 250, 0.6],
                [255, 118, 360, 235, 0.7]])
    thresh = 0.6
    keep=nms(dets,thresh)
    print(keep[0])