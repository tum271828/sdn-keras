import sdn
import util
from keras.callbacks import *

import time

basepath="/home/tum/ds/topviewcarsegbox.part"
trainImagePath=f"{basepath}/leftImg8bit/train"
trainMaskPath=f"{basepath}/gtFine/train"
testImagePath=f"{basepath}/leftImg8bit/val"
testMaskPath=f"{basepath}/gtFine/train"
batchSize=4
nImg=577
useHierarchicalSupervision=True
useScoreMapConnect=True

if useHierarchicalSupervision:
    sdn=sdn.SDN(3,useScoreMapConnect=useScoreMapConnect)
    logger=CSVLogger("log.hsv{}.csv".format(time.time()),append=True)
    g=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,6,sdn.preprocess)   
    g2=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,6,sdn.preprocess,shuffle=False)   
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
    sdn.trainModel.load_weights("weight.hsv_smap2.hdf5")
    sdn.trainModel.compile(loss=loss,loss_weights=lossWeight,optimizer="adam",
        metrics=['accuracy'])
    sdn.trainModel.fit_generator(g, nImg//batchSize, 50, callbacks=[logger,lr,stop],
        validation_data=g2, validation_steps=nImg//batchSize)
    sdn.trainModel.save_weights("weight.hsv_smap3.hdf5", overwrite=True)

else:
    sdn=sdn.SDN(3)
    logger=CSVLogger("log.no_hsv{}.csv".format(time.time()),append=True)
    g=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,1,sdn.preprocess)   
    g2=util.flowFromPath(trainImagePath,trainMaskPath,batchSize,1,sdn.preprocess,shuffle=False)   
    lr=ReduceLROnPlateau(monitor='loss',factor=0.1,patience=3,verbose=1,mode='min',cooldown=0,min_lr= 0)
    stop=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='min')
    sdn.model.compile(loss="categorical_crossentropy",optimizer="adam",
        metrics=['accuracy'])

    sdn.model.fit_generator(g, nImg//batchSize, 50,callbacks=[logger,lr,stop],
        validation_data=g2, validation_steps=nImg/batchSize)
    sdn.model.save_weights("weight.no_hsv.hdf5", overwrite=True)

