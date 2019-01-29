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

def flowFromPath(imagePath,maskPath,batchSize,nOutput):
    print(os.path.join(imagePath,"*.png"))
    imgFiles=glob.glob(os.path.join(imagePath,"*.png"))
    maskFiles=glob.glob(os.path.join(maskPath,"*.png"))
    filenames=set([os.path.basename(f) for f in imgFiles]).intersection(
        [os.path.basename(m) for m in maskFiles])
    
    #[os.path.join(maskPath,os.path.basename(f)) for f in imgFiles]
    masks=[loadMask(os.path.join(maskPath,f),3,224,224) for  f in filenames]
    images=[loadImage(os.path.join(imagePath,f),224,224) for  f in filenames]
    n=len(filenames)
    while True:
        perm=np.arange(n)
        random.shuffle(perm)  
        for i in range(n):
            p=perm[i]
            yield np.array(images[p]).reshape(1,224,224,3),[np.array(masks[p]).reshape(1,224,224,3)]*nOutput

