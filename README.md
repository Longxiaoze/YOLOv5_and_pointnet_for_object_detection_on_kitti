# 基于YOLOv5和pointnet的图像+点云的kitti自动驾驶场景目标检测

YOLOv5_and_pointnet_for_object_detection_on_kitti

## 摘要

​	在近些年得益于人工智能的快速发展，深度学习加持下的自动驾驶在这些年发展迅速，已逐渐得到越来越多的科研和社会关注。在本次项目中，我们构建了基于kitti公共自动驾驶数据集的图像和雷达点云的物体识别系统。图像目标检测任务中我们使用YOLOv5作为检测网络，在kaggle平台Tesla P100-PCIE-16GB上使用YOLOv5x(模型最大，效果最好，速度最慢)模型单张图片检测平均时间为0.044s，即22.73FPS，同时得到了很好的检测效果；雷达点云目标检测任务中，由于雷达点云数据的特殊性，我们使用算法将地面分割出去，通过聚类方法得到物体，然后通过pointnet点云分类网络将聚类得到的物体进行分类，再转换成检测框的形式，并在kitti官方评判代码中取得了不错的效果。我们还添加了可视化结果，包括但不限于将图像投影到点云，将点云投影到图像等。最终阐述了部分不足与未来的改进方向。

总代码见[github(点击进入)](https://github.com/Longxiaoze/YOLOv5_and_pointnet_for_object_detection_on_kitti)

带有部分数据集的代码见[百度网盘(点击进入)](https://pan.baidu.com/s/1tjJuhY47BHEms3uokNnvIg  )

链接: https://pan.baidu.com/s/1tjJuhY47BHEms3uokNnvIg  

密码: sda6



## 基于YOLOv5的kitti数据集2D目标检测

代码：[kaggle平台复现](https://www.kaggle.com/longxiaoze/yolov5-in-kitti-detection/notebook)

在图像端，我们使用了YOLOv5作为图像端的2D目标检测模型，使用官方在COCO数据集上的预训练模型在kitti公共数据集进行检测，通过观察实验结果，验证了其准确率和召回率，得到了较好的图像检测结果。



## 基于pointnet的点云分类网络下的目标检测

代码：[github](https://github.com/Longxiaoze/YOLOv5_and_pointnet_for_object_detection_on_kitti)

主要思路是：

​	（1）将点云地面用非深度学习方法进行分割，

​	（2）将剩余高出地面的点云进行聚类，

​	（3）将聚类后的点云放在pointnet中进行分类，

​	（4）最后将分类的点云数据进行标签转换，转换为3D目标检测框。



## 目标检测数据可视化

​	最后的部分，为了使工作以及未来数据融合更方便，我们将数据进行可视化展示，包括但不限于：

​	（1）图像和点云数据的原始数据可视化

![图像和点云数据的原始数据可视化](img_for_md/图像和点云数据的原始数据可视化.png)

​	（2）图像+2D检测框的可视化

![图像+2D检测框的可视化](img_for_md/图像+2D检测框的可视化.png)

​	（3）图像+3D检测框的可视化

![图像+3D检测框的可视化](img_for_md/图像+3D检测框的可视化.png)

​	（4）点云+3D检测框的可视化

![点云+3D检测框的可视化](img_for_md/点云+3D检测框的可视化.png)

​	（5）图像RGB值投影到点云的可视化

![图像RGB值投影到点云的可视化](img_for_md/图像RGB值投影到点云的可视化.png)

​	（6）点云投影到图像的可视化

![点云投影到图像的可视化](img_for_md/点云投影到图像的可视化.png)

​	（7）点云地面分割结果可视化

![点云地面分割结果可视化](img_for_md/点云地面分割结果可视化.png)

​	（8）点云分割结果可视化

![点云分割结果可视化](img_for_md/点云分割结果可视化.png)

​	（9）点云连续帧检测GIF可视化

![点云连续帧检测GIF可视化1](img_for_md/点云连续帧检测GIF可视化1.GIF)

![点云连续帧检测GIF可视化2](img_for_md/点云连续帧检测GIF可视化2.GIF)

  （10）视频检测GIF可视化

![视频检测GIF可视化1](img_for_md/视频检测GIF可视化1.GIF)

![视频检测GIF可视化2](img_for_md/视频检测GIF可视化2.GIF)
