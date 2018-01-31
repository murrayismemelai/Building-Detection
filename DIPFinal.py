# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:51:11 2017

@author: USER
"""
from PIL import Image
import numpy as np
import cv2
from skimage import exposure
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage import filters
from sklearn.cluster import KMeans
from skimage import img_as_ubyte
from skimage import io,data,color
from skimage import filters
from skimage import exposure
import random
import time




#ImOriginal=cv2.imread('chicago2.tif')
#CheckWrongpoint=np.array(Image.open('chicago2_total_detection.tif').convert('L'))

#ImOriginalgt=np.array(Image.open('chicago2cut_gt.tif').convert('L'))   #ground truth
ImOriginal=cv2.imread('input4cut.tif')  #Image
CheckWrongpoint=np.array(Image.open('input4_shadowcut.tif').convert('L'))  #wrongpoints

s,stride,THstd=7,10,25
box,THDist=160,15 
Openingsize,Opening2size,bilateralsize=10,10,75
huge,small,THRatio=30000,600,700   #THRatio depends on the area of building,so it is not a good way to test the strange shape area. 
area=0  # 0:Buildings are far smaller than Image size    # 1:Buildings are not far smaller than Image size
output=315


"""
ImOriginal=cv2.imread('281_0.tif')  #Image
ImInfcrop=ImOriginal[3850:4500,3550:3850]
cv2.imwrite("281_0cut3.tif",ImInfcrop) 
"""

#cut
"""
ImOriginal=cv2.imread('input2.tif')  #Image
ImInfcrop=ImOriginal[3000:3500,0:500]
cv2.imwrite("1.tif",ImInfcrop) 
img = Image.open('1.tif')
aaa=img.resize( (5000,5000), Image.BILINEAR)
aaa.save("2.png")
a=cv2.imread('2.png')  #Image
cv2.imwrite("chicago2_wrong.tif",a) 
"""

"""
img = Image.open('input5_shadow.tif')
aaa=img.resize( (1000,1000), Image.BILINEAR)
aaa.save("2.png")
a=cv2.imread('2.png')  #Image
cv2.imwrite("input5_shadowcut.tif",a) 
"""





#define function
def Findstd2(ImOriginal100,CheckWrongpoint100,s,stride):
 k=0
 size=int(ImOriginal100.shape[0]/stride)
 ImOriginalresize100=np.zeros(shape=((2*s+1)**2,3,size**2)) 
 for i in range(s,ImOriginal100.shape[0]-s,stride):
    for j in range(s,ImOriginal100.shape[0]-s,stride):
        if CheckWrongpoint100[i,j]==0:
         ImOriginalresize100[0:(2*s+1)**2,0:3,k]=ImOriginal100[i-s:i+s+1,j-s:j+s+1,:].reshape((2*s+1)**2,3)
        k=k+1         
 std=np.zeros(shape=(3,size**2)) 
 std[0,:]=np.std(ImOriginalresize100[:,0,:],axis=0)                    
 std[1,:]=np.std(ImOriginalresize100[:,1,:],axis=0)
 std[2,:]=np.std(ImOriginalresize100[:,2,:],axis=0)
 stdsum100=std[0,:]+std[1,:]+std[2,:]
 k=0
 ImOriginalstd100=np.zeros(shape=(ImOriginal100.shape[0],ImOriginal100.shape[1])) 
 for i in range(s,ImOriginal100.shape[0]-s,stride):
      for j in range(s,ImOriginal100.shape[0]-s,stride):
          ImOriginalstd100[i,j]=stdsum100[k]
          k=k+1
 return ImOriginalstd100



'''
def crop(ImInf100,a,b,c,d):
    crop100=np.zeros(shape=(b-a,d-c,ImInf100.shape[2]))
    crop100[:,:,:]=ImInf100[a:b,c:d,:]
    return crop100 
'''
  
def ImandBuilding(ImOriginal100,reference):
  ImOriginal200=np.zeros(shape=(ImOriginal100.shape[0],ImOriginal100.shape[1],3)) 
  ImOriginal200[:,:]=ImOriginal100[:,:,:]
  ImOriginal200[reference==255]=[255,255,255]
  return ImOriginal200


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
tStart = time.time()#計時開始 # RGB to Lab and find seeds

ImOriginallab=cv2.cvtColor(ImOriginal,cv2.COLOR_BGR2LAB) #ImOrigina to Lab
ImOriginallab[CheckWrongpoint==255]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)] # assign randompoint to wrongpoints
stdLabsum=Findstd2(ImOriginallab,CheckWrongpoint,s,stride)  #findstd
Checklowstd=np.zeros(shape=(ImOriginal.shape[0],ImOriginal.shape[1])) 
Checklowstd[(stdLabsum<THstd)&(stdLabsum!=0)]=[255]   #find seeds
n=int(np.sum(Checklowstd/255))  #number of seeds
print('number of seeds:%d'%(n)) # output number of seeds 
cv2.imwrite("%d_0.tif"%output,ImandBuilding(ImOriginal,Checklowstd))   #imwrite ImOriginal and seeds


tEnd = time.time()#計時結束
t1=tEnd - tStart
print('Findseed:%f' %(t1))  






tStart = time.time()#計時開始  #ImOrigina do bilateral and to lab

ImInf=np.zeros(shape=(ImOriginal.shape[0],ImOriginal.shape[1],6))  
ImInf[:,:,0:3]=ImOriginal[:,:,:]
ImInf[:,:,3]=CheckWrongpoint[:,:]
ImInf[:,:,4]=Checklowstd[:,:] 
ImInfcrop=ImInf[:,:,:]
#ImInfcrop=crop(ImInf,3000,3500,0,500)
ImInfcropbilateral = np.float64(cv2.bilateralFilter(np.float32(ImInfcrop[:,:,0:3]),7,bilateralsize,bilateralsize))   #can be negative   will be zero when output
Nor=ImInfcropbilateral.max()  #ImInfcropbilateral output may be larger than 255,  normalize
if Nor>255:
   Nor=int(Nor+1)
else:
   Nor=255   
ImInfcropbilateral=cv2.cvtColor(img_as_ubyte(ImInfcropbilateral/Nor),cv2.COLOR_BGR2LAB) 

tEnd = time.time()#計時結束
t2=tEnd - tStart
print('Bilateral and Lab:%f' %(t2)) 







tStart = time.time()#計時開始  # find Dist  

IFbuilding2=np.ones(shape=(box,(box+5)*n,3))*1000 
keypoint=np.zeros(shape=(3,n)) 
keypoint2=np.zeros((6,n),np.int) 
ImInfcropbilateralwrong=np.zeros(shape=(ImInfcropbilateral.shape[0],ImInfcropbilateral.shape[1],3)) 
ImInfcropbilateralwrong[:,:,:]=ImInfcropbilateral[:,:,0:3]
ImInfcropbilateralwrong[ImInfcrop[:,:,3]==255]=1000
k=0   
for j in range(s,ImInfcrop.shape[1],stride):    #map to new matrix 
    for i in range(s,ImInfcrop.shape[0],stride):
        if ImInfcrop[i,j,4]==255:             
              L,R,U,B=int(box/2),int(box/2),int(box/2),int(box/2)  #人為定義
              if i-U<0:
                  U=i
              if i+B>=ImInfcrop.shape[0]:
                  B=ImInfcrop.shape[0]-i 
              if j-L<0:
                  L=j
              if j+R>=ImInfcrop.shape[1]:
                  R=ImInfcrop.shape[1]-j
              IFbuilding2[0:(U+B),(box+5)*k:(box+5)*k+(L+R),:]=ImInfcropbilateralwrong[i-U:i+B,j-L:j+R,:]              
              keypoint[0:3,k]=ImInfcropbilateralwrong[i,j,:]
              keypoint2[0:6,k]=j,i,L,R,U,B        
              k+=1 
IFbuilding2tall=IFbuilding2.transpose((1,0,2)).reshape(n,box*(box+5),3).transpose((1,0,2))  
Disttall = np.linalg.norm(IFbuilding2tall - np.transpose(keypoint[0:3,:]),axis=2)           
Dist=np.transpose(np.transpose(Disttall).reshape((box+5)*n,box))

tEnd = time.time()#計時結束
t3=tEnd - tStart
print('find distance with seed:%f' %(t3))  







tStart = time.time()#計時開始   # find building, oopenng , connectedComponent 

kernelOpening = np.ones((Openingsize,Openingsize),np.uint8)
#buildingNumber=np.zeros(shape=(int((THhigh-THlow)/THstride),n)) 
building=np.zeros(shape=(box,(box+5)*n))               
building[Dist<THDist]=1
building[Dist>THDist]=0
#buildingreshape=np.transpose(np.transpose(building).reshape((box+5)*n,box))
buildingOpening = cv2.morphologyEx(building, cv2.MORPH_OPEN, kernelOpening)
buildingOCc = cv2.connectedComponentsWithStats(img_as_ubyte(buildingOpening), 4, cv2.CV_32S)
ImInfcrop[:,:,5]=np.zeros(shape=(ImInfcrop.shape[0],ImInfcrop.shape[1]))
for kk in range(0,n): 
  if buildingOCc[1][keypoint2[4,kk],kk*(box+5)+keypoint2[2,kk]]!=0:   #The low std points may be cut by opening
    aa=buildingOCc[1][0:keypoint2[4,kk]+keypoint2[5,kk],kk*(box+5):kk*(box+5)+keypoint2[2,kk]+keypoint2[3,kk]]  #map building points to Imoriginal
    ImInfcrop[keypoint2[1,kk]-keypoint2[4,kk]:keypoint2[1,kk]+keypoint2[5,kk],keypoint2[0,kk]-keypoint2[2,kk]:keypoint2[0,kk]+keypoint2[3,kk],5][aa==buildingOCc[1][keypoint2[4,kk],kk*(box+5)+keypoint2[2,kk]]]+=1 
    #buildingNumber[k,kk]=buildingOCc[2][buildingOCc[1][keypoint2[4,kk],kk*(box+5)+keypoint2[2,kk]],4]   #record numbers of building points  
ImInfcrop[:,:,5][ImInfcrop[:,:,5]!=0]=[255]
cv2.imwrite("%d_1.tif"%output,ImandBuilding(ImInfcrop[:,:,0:3],ImInfcrop[:,:,5]))

tEnd = time.time()#計時結束
t4=tEnd - tStart
print('findBuilding:%f' %(t4)) 





tStart = time.time()#計時開始  # opening, cut huge and smal area, cut strange building by contour

kernelOpening2 = np.ones((Opening2size,Opening2size),np.uint8)
ImOriginalbuilding=np.zeros(shape=(ImInfcrop.shape[0],ImInfcrop.shape[1])) 
ImOriginalbuilding[:,:]=ImInfcrop[:,:,5]
ImOriginalbuilding[ImInfcrop[:,:,5]!=0]=255
ImOriginalbuildingOpening = cv2.morphologyEx(ImOriginalbuilding, cv2.MORPH_OPEN, kernelOpening2)
ImCC=cv2.connectedComponentsWithStats(img_as_ubyte(ImOriginalbuildingOpening/255), 4, cv2.CV_32S)
ImCC255=np.zeros(shape=(ImCC[1].shape[0],ImCC[1].shape[1])) 
ImCC255[ImCC[1]!=0]=255
Docontour= cv2.findContours(img_as_ubyte(ImCC255/255),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
contournumber=np.zeros(shape=(ImCC[0],1)) 



if area==0:
  for i in range(0,ImCC[0]):    # small building
    if (ImCC[2][i,4]<small)|((ImCC[2][i,4]>huge)&(ImCC[2][i,4]!=ImCC[2][:,4].max())): 
            ImCC[1][ImCC[1]==i]=0
    else: 
        contournumber[i]=np.sum((ImCC[1]==i)&(Docontour[0]!=1)) 
  contournumber[contournumber==0]=1000000            
  contourareaRatio=contournumber/ImCC[2][:,4:5]**(1/2)
  ImCCcontour=np.zeros(shape=(ImCC[1].shape[0],ImCC[1].shape[1])) 
  ImCCcontour[:,:]=ImCC[1][:,:]
  for i in range(0,ImCC[0]): 
    if (contourareaRatio[i]>THRatio):        #big bugs
            ImCCcontour[ImCCcontour==i]=0



if area==1:
  for i in range(0,ImCC[0]):    # big building
    if (ImCC[2][i,4]<small): 
         ImCC[1][ImCC[1]==i]=0
  ImCCcontour=ImCC[1]




tEnd = time.time()#計時結束
t5=tEnd - tStart
print('cut strange building:%f' %(t5)) 







tStart = time.time()#計時開始  #tet data
#test
ImCC[1][ImCC[1]!=0]=255        
cv2.imwrite("%d_2.tif"%output,ImandBuilding(ImInfcrop[:,:,0:3],ImCC[1]))
ImCCcontour[ImCCcontour!=0]=255
cv2.imwrite("%d_3.tif"%output,ImandBuilding(ImInfcrop[:,:,0:3],ImCCcontour))

#ImOriginalgtcc=cv2.connectedComponentsWithStats(ImOriginalgt, 4, cv2.CV_32S)
#Result=ImCCcontour
#FinaBnumber=cv2.connectedComponentsWithStats(img_as_ubyte(Result/255), 4, cv2.CV_32S)
#ImOriginalgtandme=np.zeros(shape=(ImOriginalgt.shape[0],ImOriginalgt.shape[1])) 
#ImOriginalgtandme[ImOriginalgt==Result]=[255]
#print('acc:%f'%(np.sum(ImOriginalgtandme/255)/(5000**2))) 
#print ('BildingNumderGT:%d'%(ImOriginalgtcc[0]))
#print('BildingNumderMe:%d'%(FinaBnumber[0]))
#print('GTandMeRatio:%f'%((FinaBnumber[0]-ImOriginalgtcc[0])/ImOriginalgtcc[0]))
    
tEnd = time.time()#計時結束
t6=tEnd - tStart
print('test:%f' %(t6))  
print('total time:%f minutes' %((t1+t2+t3+t4+t5+t6)/60))  



#resize output image to 5000X5000 
cv2.imwrite("1.tif",ImCC[1]) 
img = Image.open('1.tif')
aaa=img.resize( (5000,5000), Image.BILINEAR)
aaa.save("2.png")
a=cv2.imread('2.png')  #Image
cv2.imwrite("outputtest.tif",a) 
