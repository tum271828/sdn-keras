import cv2,glob,os,random
import numpy as np

def loadMask(path, nClass, width, height):
    mask = np.zeros((height, width, nClass),dtype=np.uint8)
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]

    for c in range(nClass):
        mask[:, :, c] = (img == c).astype(int)
    return mask

def loadImage(path, width, height):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    return img

def flowFromPath(imagePath,maskPath,batchSize,nOutput,preprocess,shuffle=True):
    imgFiles=glob.glob(os.path.join(imagePath,"*.png"))
    maskFiles=glob.glob(os.path.join(maskPath,"*.png"))
    filenames=set([os.path.basename(f) for f in imgFiles]).intersection(
        [os.path.basename(m) for m in maskFiles])
    masks=[loadMask(os.path.join(maskPath,f),3,224,224) for  f in filenames]
    images=[loadImage(os.path.join(imagePath,f),224,224) for  f in filenames]
    n=len(filenames)
    perm=np.arange(n)    
    X=[]
    Y=[]
    while True:
        if shuffle:
            random.shuffle(perm)
        for i in range(n):
            p=perm[i]
            X.append(preprocess(images[p]))
            Y.append(masks[p])
            if len(X)==batchSize:
                yield np.array(X),[np.array(Y)]*nOutput
                X=[]
                Y=[]

