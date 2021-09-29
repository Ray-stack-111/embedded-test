import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('/home/loongson/Desktop/output/output01=.avi', fourcc, 20.0, (640,  360))

cap = cv2.VideoCapture('/home/loongson/Desktop/output/road.flv')
if not cap.isOpened():
    print("camera shutdown")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("video over")
        break

    img1=frame
    img2=frame
    #读取图片
        
    bgr= [100,150 ,200 ]
    thresh = 50
    #指定黄色区域的相关范围，试图使用BGR通道进行颜色分离
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
    #制作mask，将所有非黄色区域变成黑色
    maskBGR = cv2.inRange(img1,minBGR,maxBGR)
    img1 = cv2.bitwise_and(img1,img1, mask = maskBGR)
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
     
    img1=cv2.medianBlur(img1,5) 
    #使用中值滤波去除噪点
    
    kernel=np.ones((3,3),np.uint8)
    img1=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel)
    ret, img1=cv2.threshold(img1,50,255,cv2.THRESH_BINARY)
    #进行开操作，将图像的毛边等弱化，之后将其转换为二值图像
    
    _,contours,hier =cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    draw_img=img2.copy()
    res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
    #找出轮廓

    n = len(contours)  
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 4000 :
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
        
    #遍历所有轮廓，将其中面积较小的部分变成黑色
    img1=cv2.fillPoly(img1, cv_contours, (0, 0, 0))
    
    _,contours,hier =cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    draw_img=img2.copy()
    res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
    n = len(contours)  
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 4000 :
            cv_contours.append(contour)
            # x, y, w, h = cv2.boundingRect(contour)
            # img[y:y + h, x:x + w] = 255
        else:
            continue
#与上一个遍历相反，意在将图像中的孔洞堵住
    img1=cv2.fillPoly(img1, cv_contours, (255, 255, 255))
    
    kernel=np.ones((3,3),np.uint8)
    img1=cv2.dilate(img1,kernel,iterations=2)
    #进行膨胀操作
    
    _,contours,hier =cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    draw_img=img2.copy()
    res=cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
    #画出轮廓
    b=np.hstack((img2,res))
    cv2.imshow('img',b)
    #将两张图像放置在一起进行对比
    out.write(res)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
