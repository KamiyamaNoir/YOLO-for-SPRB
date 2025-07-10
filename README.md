# YOLO on SummerPockets PingPong Game

随便玩玩...
提供二次曲线拟合和卡尔曼滤波预测，脚本中默认采用卡尔曼滤波预测

## 训练集

使用VoTT标注，导出为csv格式，然后使用`distribute.py`来转换为YOLO可识别的格式

## 二次曲线参数方程形式与点距离数学算法

对于由参数方程  
$x = At^2+Bt+C$  
$y = Dt^2+Et+F$  
构成的二次曲线  
一点$P(x,y)$到该曲线的距离的求解方式：  
$(2A^2+2D^2)t^3+(3AB+3DE)t^2+(2DF+E^2+2AC+B^2-2Dy-2Ax)t+B(C-x)+E(F-y)=0$
