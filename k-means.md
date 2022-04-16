1、首先我们要知道我们需要聚类的是bounding box，所以我们无需考虑其所属类别，
我们将所有的bounding box坐标提取出来，也许一张图有一个矩形框，也许有多个，但是我们需要无区别的将所有图片的矩形框提取出来，放在一起

2、数据处理：获得所有训练数据bounding box的宽高大小，所以我们需要将坐标数据转换为框的宽高大小
长=右小角横坐标-左上角横坐标
宽=右下角纵坐标-左上角纵坐标

3、初始化k个anchor box,通过在所有的bounding boxex中随机选取k个值作为k个anchor boxes的初始值。

4、计算每个bounding box与每个anchor box的iou值。传统的聚类方法是使用欧氏距离来衡量差异，也就是说如果我们运用传统的k-means聚类算法，可以直接聚类bounding box的宽和高，产生K个宽、高组合的anchor boxes,但是作者发现此方法在box尺寸比较大的时候，其误差也更大，所以作者引入了iou值，可以避免这个问题。

min_w_matrix=np.minimum(cluster_w_matrix,box_w_matrix) 分别是anchor和其它bounding box的宽度
min_h_matrix=np.minimum(cluster_h_matrix,box_h_matrix) 分别是anchor和其它bounding box的高度
iter_area=np.multiply(min_w_matrix,min_h_matrix)
IOU=iter_area/(box_area+cluster_area-iter_area)

由于iou值越大越好,所以定义一个距离d参数，用来表示其误差：d=1-iou

5、分类操作
经过前一步的计算可以得到每一个bounding box对于每一个anchor box的误差d(n,k)，我们通过比较每个bounding box其对于每个anchor box的误差大小{d(i,1),...,d(i,k)},选取最小误差的那个anchor box，将这个bounding box分类给它，对于每个bounding box都做这个操作，最后记录下来每个anchor box有哪些bounding box属于它。

6、anchor box更新
经过上一步，我们知道每个anchor box都有哪些bounding box属于它，然后对于每个anchor box中的那些bounding box，我们再求这些bounding box的宽高中值大小，将其作为该anchor box新尺寸。

7、重复4-6步骤，直到在第五步中发现对于全部bounding box其所属的anchor box类与之前所属的anchor box类完全一样。（这里表示所有bounding box的分类已经不再更新）

8、计算anchor boxes精确度。至第七步，其实已经通过k-means算法计算出anchor box。但是我们还是可以计算精确度大小，
计算方法：使用最后得到的anchor boxes与每个bounding box计算其IOU值，对于每个bounding box选取其最高的那个IOU值（其实代表了属于某一个anchor box类），然后求所有bounding box的IOU平均值，也就是精确度值。也就是计算平均IOU

code

import numpy as np

#这里核心使用的是广播机制

def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def k_means(boxes, k, dist=np.median):
    """
    yolo k-means methods
    refer: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    Args:
        boxes: 需要聚类的bboxes
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))

    # 在所有的bboxes中随机挑选k个作为簇的中心。
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
    	# 计算每个bboxes离每个簇的距离 1-IOU(bboxes, anchors)
        distances = 1 - wh_iou(boxes, clusters)
        
        # 计算每个bboxes距离最近的簇中心
        current_nearest = np.argmin(distances, axis=1) #返回的是每个bounding box对应的anchor的 下标
        
        # 每个簇中元素不在发生变化说明以及聚类完毕
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # 根据每个簇中的bboxes重新计算簇中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters

# np.median使用
a:输入的数组
axis：计算轴
out：存储求取中位数后的数组
overwrite_input:bool值，表示是否在原数组内计算
keepdims:求取中位数的那个轴将保留在结果中

median(a, axis=None, out=None,overwrite_input=False, keepdims=False)

>>> a = np.array([[10, 7, 4], [3, 2, 1]])
>>> a
array([[10,  7,  4],
       [ 3,  2,  1]])
>>> np.median(a)
3.5
>>> np.median(a, axis=0)
array([ 6.5,  4.5,  2.5])
>>> np.median(a, axis=1)
array([ 7.,  2.])
>>> m = np.median(a, axis=0)
>>> out = np.zeros_like(m)
>>> np.median(a, axis=0, out=m)
array([ 6.5,  4.5,  2.5])
>>> m
array([ 6.5,  4.5,  2.5])
>>> b = a.copy()
>>> np.median(b, axis=1, overwrite_input=True)
array([ 7.,  2.])
>>> assert not np.all(a==b)
>>> b = a.copy()
>>> np.median(b, axis=None, overwrite_input=True)
3.5