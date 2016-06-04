import Image
import numpy as np

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


Wprocent=np.array([  6.63885766e-01  ,-2.98026395e-05  ,-3.91967890e-05])
a=np.array([[0.257,0.504,0.098],
[-0.148,-0.291, 0.439],
[0.439,-0.368,-0.071]])
b=np.array([
16,
128,
128,
])

I=np.load("/home/laurentprat/data/facedetect/I.npy")
data=np.load("/home/laurentprat/data/facedetect/data2.npy")
skin=np.ones((15,44,44,3))
skin[0,:,:,:]=np.asarray(Image.open("skin/skinstandart.jpg"),dtype="int32")
for i in range(1,13):
    skin[i,:,:,:]=np.asarray(Image.open("skin/skin"+str(i+1)+".jpg"),dtype="int32")

skin[13, :, :, :] = np.asarray(Image.open("skin/skin" + str(33 + 1) + ".jpg"), dtype="int32")
skin[14, :, :, :] = np.asarray(Image.open("skin/skin" + str(44 + 1) + ".jpg"), dtype="int32")
skin=skin.dot(np.transpose(a))+b
r =28
img = Image.open("test_images/" + str(r) + ".jpg")
img = np.asarray(img, dtype="int32")
nimg=np.ones((img.shape[0],img.shape[1],3))
nimg=toycbcr(img,a,b)
x=np.ones((3))
x[1]=img.shape[0]
x[2]=img.shape[1]
simageall = np.ones((img.shape[0], img.shape[1], 3, 14))
nimage=np.zeros((img.shape[0],img.shape[1]))

ns=13
maxi=00
procent=np.matmul(x,Wprocent)
coeff=1
while maxi<img.shape[0]*img.shape[1]*3*procent:

    for i in range(0,15):
        medm0 = np.median(skin[i, :, :, 0]) - np.std(skin[i, :, :, 0])*coeff
        medp0 = np.median(skin[i, :, :, 0]) + np.std(skin[i, :, :, 0]) * 4.5*coeff
        medm1 = np.median(skin[i, :, :, 1]) - np.std(skin[i, :, :, 1]) * 0.8*coeff
        medp1 = np.median(skin[i, :, :, 1]) + np.std(skin[i, :, :, 1]) * 0.4*coeff
        medm2 = np.median(skin[i, :, :, 2]) - np.std(skin[i, :, :, 2]) * 0.4*coeff
        medp2 = np.mean(skin[i, :, :, 2]) + np.std(skin[i, :, :, 2])*coeff

        inter=nimg
        sum=np.sum(np.abs(nimg[:,:,0]-(medm0+medp0)/2)<medp0-(medm0+medp0)/2)+np.sum(np.abs(nimg[:,:,1]-(medm1+medp1)/1)<medp1-(medm1+medp1)/1)+np.sum(np.abs(nimg[:,:,2]-(medm2+medp2)/2)<medp2-(medm2+medp2)/2)
        if sum>maxi:
            maxi=sum
            ns=13
    coeff=coeff+1


interimage=nimg[:,:,0]
interimage1=nimg[:,:,1]
interimage2=nimg[:,:,2]

interimage[(interimage<medm0)]=0
interimage[(interimage>medp0)]=0

interimage1[interimage==0]=0
interimage1[(interimage1<medm1)]=0
interimage1[(interimage1>medp1)]=0

interimage2[interimage1==0]=0
interimage2[(interimage2<medm2)]=0
interimage2[(interimage2>medp2)]=0
interimage2[(interimage2!=0)]=255
print np.sum(interimage2==255)
print nimg

im=Image.fromarray(interimage2.astype(np.uint8))
im.save("results/skindetection.jpg")
i2=0
n=40
plus=0
minus=0
index=0
mean=np.mean(interimage2)
howmany=0
index_horizontal=np.ones((20))
index_size=np.ones((20))
plus =0
minus=0
for i2 in range(0,img.shape[1],40):
    if np.sum(interimage2[:,i2 - 20:i2 + 20]) > img.shape[0] * 40*255/8 :
        plus =plus+1
        sum = sum+np.sum(interimage2[:, i2 - 20:i2 + 20])



    else:
        if plus!=0:

                index_horizontal[howmany] = i2-(plus/2+minus)*40
                index_size[howmany]=sum
                howmany = howmany + 1
                plus=0
                minus=0
                sum=0




twobest=np.zeros((howmany))
index_horizontal[howmany]=img.shape[1]
index_horizontal=index_horizontal[0:howmany+1]
twobestindex=np.zeros((howmany))
index_size=index_size[0:howmany]
meansize=np.mean(index_size)

for i in range(20,img.shape[0],40):
    for index in range(0,howmany):
      if index_size[index]>meansize/2:
        if twobest[index]<np.sum(interimage2[i - 20:i + 20,index_horizontal[index]-60:index_horizontal[index+1]-60]):
            twobest[index]=np.sum(interimage2[i - 20:i + 20,index_horizontal[index]-60:index_horizontal[index+1]-60])
            twobestindex[index]=i
n=40
procent=img.shape[0]*img.shape[1]/100000


for index in range(0,howmany):
 if index_size[index] > meansize / 2:

    plus = 0
    minus = 0
    top = 0
    bottom = 0
    i=twobestindex[index]
    i2=index_horizontal[index]

    num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
    while num>n*n/procent:
        i=i-n
        num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
        minus=minus+1
    i=i+minus*n
    num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)


    while num > n * n / procent:
        i = i + n
        num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
        plus=plus+1
    i=i-plus*n
    num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
    while num > n * n / procent:
        i2 = i2 - n
        num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
        bottom = bottom + 1
    i2=i2+bottom*n

    num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
    while num > n * n / procent:
        i2 = i2 + n
        num = np.sum(interimage2[i:i + n, i2:i2 + n] == 255)
        top=top+1
    i2=i2-top*n
    img[i-minus*n:i-minus*n+10 , i2-bottom*n:i2 + top*n, 0] = 255
    img[i-minus*n:i-minus*n +10, i2-bottom*n:i2 + top*n, 1] = 0
    img[i-minus*n:i+-minus*n +10, i2-bottom*n:i2 + top*n, 2] = 0

    img[i+plus*n :i  + plus * n+10, i2 - bottom * n:i2 +top * n+10, 0] = 255
    img[i +plus*n:i + plus * n+10, i2 - bottom * n:i2 + top * n+10, 1] = 0
    img[i +plus*n:i +  plus * n+10, i2 - bottom * n:i2 + top* n+10, 2] = 0

    img[i - minus * n:i+plus*n , i2 - bottom * n:i2 -bottom*n+10, 0] = 255
    img[i - minus * n:i+plus*n , i2 - bottom * n:i2 -bottom*n+10, 1] = 0
    img[i - minus * n:i+plus*n , i2 - bottom * n:i2 -bottom*n+10, 2] = 0

    img[i - minus * n:i+plus*n , i2 + top * n:i2 +top*n+10, 0] = 255
    img[i - minus * n:i+plus*n , i2 + top * n:i2 +top*n+10, 1] = 0
    img[i - minus * n:i+plus*n , i2 + top * n:i2 +top*n+10, 2] = 0

    img[i-5:i+5,i2-5:i2+5,:]=255

im=Image.fromarray(img.astype(np.uint8),"RGB")
im.save("results/facedetection.png")



