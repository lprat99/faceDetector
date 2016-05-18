import os,sys
import Image
import numpy as np
import random
import tensorflow as tf
def torgb(img3):
    img2=np.ones(img.shape)
    img2[:, :, 0]= 1.164*(img3[:,:,0] - 16) + 1.596*(img3[:,:,2] - 128)

    img2[:, :, 1]= 1.164*(img3[:,:,0] - 16) - 0.813*(img3[:,:,2] - 128) - 0.392*(img3[:,:,1] - 128)

    img2[:, :, 2]= 1.164*(img3[:,:,0] - 16) + 2.017*(img3[:,:,1] - 128)
    return img2
def toycbcr(img,a,b):
    nimg=np.zeros((img.shape))
    nimg[:,:,0]=img[:,:,0]*a[0,0]+img[:,:,1]*a[0,1]+img[:,:,2]*a[0,2]+b[0]
    nimg[:,:,1]=img[:,:,0]*a[1,0]+img[:,:,1]*a[1,1]+img[:,:,2]*a[1,2]+b[1]
    nimg[:,:,2]=img[:,:,0]*a[2,0]+img[:,:,1]*a[2,1]+img[:,:,2]*a[2,2]+b[2]
    return nimg

a=np.array([[0.257,0.504,0.098],
[-0.148,-0.291, 0.439],
[0.439,-0.368,-0.071]])
b=np.array([
16,
128,
128,
])



skin=np.ones((15,44,44,3))
skin[0,:,:,:]=np.asarray(Image.open("skin/skinstandart.jpg"),dtype="int32")
for i in range(1,13):
    skin[i,:,:,:]=np.asarray(Image.open("skin/skin"+str(i+1)+".jpg"),dtype="int32")

skin[13, :, :, :] = np.asarray(Image.open("skin/skin" + str(33 + 1) + ".jpg"), dtype="int32")
skin[14, :, :, :] = np.asarray(Image.open("skin/skin" + str(44 + 1) + ".jpg"), dtype="int32")

skin=skin.dot(np.transpose(a))+b

r = 65
img = Image.open("test_images/" + str(r) + ".jpg")
img = np.asarray(img, dtype="int32")
nimg=np.ones((img.shape[0],img.shape[1],3))
nimg=toycbcr(img,a,b)
simageall = np.ones((img.shape[0], img.shape[1], 3, 14))
nimage=np.zeros((img.shape[0],img.shape[1]))
coeff=6
ns=13
mini=100000000000000000000
for i in range(0,15):
    if np.sum(abs(nimg[:,:,0]-np.median(skin[i,:,:,1])))+np.sum(abs(nimg[:,:,1]-np.median(skin[i,:,:,1])))+np.sum(abs(nimg[:,:,2]-np.median(skin[i,:,:,2])))<mini:
        mini=np.sum(abs(nimg[:,:,0]-np.median(skin[ns,:,:,0])))+np.sum(abs(nimg[:,:,1]-np.median(skin[ns,:,:,1])))+np.sum(abs(nimg[:,:,2]-np.median(skin[ns,:,:,2])))
        ns=i
print ns
medm0=np.median(skin[ns,:,:,0])-np.std(skin[ns,:,:,0])*coeff
medp0=np.median(skin[ns,:,:,0])+np.std(skin[ns,:,:,0])*coeff*4.5
medm1=np.median(skin[ns,:,:,1])-np.std(skin[ns,:,:,1])*coeff*0.8
medp1=np.median(skin[ns,:,:,1])+np.std(skin[ns,:,:,1])*coeff*0.4
medm2=np.median(skin[ns,:,:,2])-np.std(skin[ns,:,:,2])*coeff*0.4
medp2=np.mean(skin[ns,:,:,2])+np.std(skin[ns,:,:,2])*coeff
for i in range (img.shape[0]):
    for i2 in range(img.shape[1]):
        if  medm0<nimg[i,i2,0] < medp0 and medm1<nimg[i,i2,1] <medp1 and medm2<nimg[i,i2,2]<medp2 :
                nimage[i,i2]=255


n=200

im=Image.fromarray(nimage.astype(np.uint8))
im.save("results/skindetection.jpg")
indices=np.zeros((2)).astype(int)
for i in range(0, img.shape[0] - n, 30):
    for i2 in range(0, img.shape[1] - n, 30):

        number = np.sum(nimage[i:i + n, i2:i2 + n]==255)

        if number > ((n*n)/2.510):
            print "number ",number
            max = number
            indices[0] = i
            indices[1] = i2

            img[i:i + 10, i2:i2 + n, 0] = 255
            img[i:i + 10, i2:i2 + n, 1] = 0
            img[i:i + 10, i2:i2 + n, 2] = 0

            img[i + n:i + n + 10, i2:i2 + n, 0] = 255
            img[i + n:i + n + 10, i2:i2 + n, 1] = 0
            img[i + n:i + n + 10, i2:i2 + n, 2] = 0

            img[i:i + n, i2:i2 + 10, 0] = 255
            img[i:i + n, i2:i2 + 10, 1] = 0
            img[i:i + n, i2:i2 + 10, 2] = 0

            img[i:i + n, i2 + n:i2 + 10 + n, 0] = 255
            img[i:i + n, i2 + n:i2 + 10 + n, 1] = 0
            img[i:i + n, i2 + n:i2 + 10 + n, 2] = 0


i = indices[0]
i2 = indices[1]




im=Image.fromarray(img.astype(np.uint8),"RGB")
im.save("results/facedetection.png")