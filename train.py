import sdn
import util
from keras.callbacks import *

sdn=sdn.SDN(3)

useHierarchicalSupervision=False
if useHierarchicalSupervision:
    logger=CSVLogger("log.hsv.csv",append=True)
    g=util.flowFromPath("/your-path/train_img",
    "/your-path/train_mask",1,6)   
    g2=util.flowFromPath("/your-path/test_img",
    "/your-path/test_mask",1,6)   
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

    sdn.trainModel.fit_generator(g, 398, 25,callbacks=[logger,lr,stop],validation_data=g2, validation_steps=398)
    sdn.trainModel.save_weights("weight.hsv.hdf5", overwrite=True)
else:
    logger=CSVLogger("log.no_hsv.csv",append=True)
    g=util.flowFromPath("/your-path/train_img",
    "/your-path/train_mask",1,6)   
    g2=util.flowFromPath("/your-path/test_img",
    "/your-path/test_mask",1,6)   
    lr=ReduceLROnPlateau(monitor='loss',factor=0.1,patience=3,verbose=1,mode='min',cooldown=0,min_lr= 0)
    stop=EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='min')
    sdn.model.compile(loss="categorical_crossentropy",optimizer="adam",
            metrics=['accuracy'])

    sdn.model.fit_generator(g, 398, 25,callbacks=[logger,lr,stop],validation_data=g2, validation_steps=398)
    sdn.model.save_weights("weight.no_hsv.hdf5", overwrite=True)