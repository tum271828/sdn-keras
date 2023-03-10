import sdn
import util
from keras.callbacks import *
import glob 
import time
import cv2 
import vabcl as bcl 
from tqdm import tqdm 

basepath="/home/tum/ds/topviewcarsegbox.bobo3a1"
trainImagePath=f"{basepath}/leftImg8bit/train"
trainMaskPath=f"{basepath}/gtFine/train"
testImagePath=f"{basepath}/leftImg8bit/val"
testMaskPath=f"{basepath}/gtFine/train"
batchSize=4
nImg=577
useHierarchicalSupervision=True
useScoreMapConnect=True

videoFn="/home/tum/data/bobo3/a1/video/DJI_0056.MP4"

if useHierarchicalSupervision:
    sdn=sdn.SDN(3,useScoreMapConnect=useScoreMapConnect)
    logger=CSVLogger("log.hsv{}.csv".format(time.time()),append=True)
    #g=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,6,sdn.preprocess)   
    #g2=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,6,sdn.preprocess,shuffle=False)   
    lr=ReduceLROnPlateau(monitor='output_loss',factor=0.1,patience=3,verbose=1,mode='min',cooldown=0,min_lr= 0)
    stop=EarlyStopping(monitor='output_loss', min_delta=0, patience=5, verbose=0, mode='min')
    loss={'output': "categorical_crossentropy", 
        'softmax_0_0': 'categorical_crossentropy','softmax_0_1': 'categorical_crossentropy',
        'softmax_1_0': 'categorical_crossentropy',
        'softmax_2_0': 'categorical_crossentropy','softmax_2_1': 'categorical_crossentropy'}
    lossWeight={'output': 1.,
        'softmax_0_0': 1,'softmax_0_1': 1,
        'softmax_1_0': 1,
        'softmax_2_0': 1,'softmax_2_1': 1}
    sdn.trainModel.compile(loss=loss,loss_weights=lossWeight,optimizer="adam",
        metrics=['accuracy'])
    sdn.trainModel.load_weights("weight.hsv_smap3.hdf5")
    v=bcl.openAny(videoFn)
    vout=bcl.VideoWriter("out.mp4",frameSplit=50)
    patchSize=224
    W,H=1920,1080
    for img in tqdm(v):
        img=cv2.resize(img,(1920,1080))
        img2=np.zeros_like(img)
        for startX in range(0,W-1,patchSize): 
            for startY in range(0,H-1,patchSize):
                toX=min(startX+patchSize,W)
                toY=min(startY+patchSize,H)
                bimg=img[startY:toY,startX:toX]
                pre=sdn.preprocess(bimg)
                mask=sdn.predict(pre)
                bimg2=sdn.showResult(bimg,mask)
                img2[startY:toY,startX:toX]=bimg2
        vout.write(img2)
    vout.close()