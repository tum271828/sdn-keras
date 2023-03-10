from classification_models.keras import Classifiers
import keras.backend as K
from keras.models import Model
from keras.layers import *
import tensorflow as tf
from functools import lru_cache 
import numpy as np 
import cv2 

@lru_cache
def buildPalette(numClass):
    np.random.seed(42)
    return np.random.randint(
        0, 255, size=(numClass, 3))

def colorlize(mask,numClass=80):
    if numClass is None:
        numClass=np.max(mask)+1
    palette=buildPalette(numClass)
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        img[mask == label, :] = color
    return img

class SDN(object):

    def __init__(self,nClass,featureNames=["conv2_block5_0_relu","conv3_block9_0_relu","conv4_block26_0_relu"],dropoutRate=0.0,weights="imagenet",useScoreMapConnect=False,height=224,width=224):        
        self.dropoutRate=dropoutRate
        self.nUpConvo=2
        self.nDownConvo=2
        self.activation="relu"
        self.nClass=nClass
        self.softmaxLayers=[]
        self.E=[None]*4*3
        self.width=width
        self.height=height
        self.useScoreMapConnect=useScoreMapConnect
        classifier, preprocess_input = Classifiers.get('densenet169')
        self.preprocess=lambda x:preprocess_input(x*255)
        basemodel = classifier((height, width, 3), weights=weights)
        outputs=list(map(lambda x:basemodel.get_layer(x).output,featureNames))
        encoderModel=Model(inputs=[basemodel.input],outputs=outputs)

        input=Input(shape=(height,width,3))
        x=input     
        self.bigFeature,self.smallFeature,x=encoderModel(x)
        x=self.upBlock(x,0,0)
        x=self.upBlock(x,0,1)
        x=self.downBlock(x,1,1)
        x=self.downBlock(x,1,0)
        x=self.upBlock(x,2,0)
        x=self.upBlock(x,2,1)
        x=self.upBlock(x,3,2,compress=False)
        x=self.upBlock(x,3,2,compress=False)
        x=Conv2D(nClass, (3,3), kernel_initializer='he_normal', padding='same')(x)
        x = Activation('softmax',name="output")(x)
        self.softmaxLayers.append(x)        
        self.trainModel=Model(input,self.softmaxLayers)
        self.trainModel.summary()
        self.model=Model(input,x)
        self.model.summary()
            
    def dropout(self,x):
        if self.dropoutRate>0.0:
            x = Dropout(self.dropoutRate)(x)
        return x

    '''
    blockTypeId: 0:small feature concat ,1: big feature concat, 2: no feature and skip path
    levelId:0+
    '''
    def upBlock(self,x,levelId,blockTypeId,compress=True):
        nFilter=48
        if blockTypeId==0:
            self.lastSmallFeature=x
        elif blockTypeId==1:
            self.lastBigFeature=x        
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Activation(self.activation)(x)
        x = Conv2DTranspose(nFilter, (4,4), strides=2, kernel_initializer='he_normal', padding='same')(x)
        x=self.dropout(x)

        if blockTypeId==0:
            x = Concatenate()([x, self.smallFeature])
        elif blockTypeId==1:
            x = Concatenate()([x, self.bigFeature])

        for _ in range(self.nUpConvo):
            skipX=x
            x = BatchNormalization(epsilon=1e-5)(x)
            x = Activation(self.activation)(x)
            x = Conv2D(nFilter, (3,3), kernel_initializer='he_normal', padding='same')(x)
            if blockTypeId!=2:
                x=self.dropout(x)                
                x = Concatenate()([skipX,x])
        if compress:
            x = self.compression(x,levelId,blockTypeId,True)
        return x

    def downBlock(self,x,levelId,blockTypeId,compress=True):
        nFilter=48
        x = MaxPooling2D()(x)

        if blockTypeId==0:
            x = Concatenate()([x, self.lastSmallFeature])
        elif blockTypeId==1:
            x = Concatenate()([x, self.lastBigFeature])

        for _ in range(self.nDownConvo):
            skipX=x
            x = BatchNormalization(epsilon=1e-5)(x)
            x = Activation(self.activation)(x)
            x = Conv2D(nFilter, (3,3), kernel_initializer='he_normal', padding='same')(x)
            if blockTypeId!=2:
                x=self.dropout(x)                
                x = Concatenate()([skipX,x])
        if compress:
            x = self.compression(x,levelId,blockTypeId,False)

        return x

    def compression(self, x,levelId,blockTypeId,up):        
        if up:
            if blockTypeId==0:
                nFilter=768
            elif blockTypeId==1:
                nFilter=576
        else:
            if blockTypeId==0:
                nFilter=1024
            elif blockTypeId==1:
                nFilter=768
        x = Activation(self.activation)(x)
        x = Conv2D(nFilter, (3,3), kernel_initializer='he_normal', padding='same')(x)        
        output=x        
        if up or blockTypeId==0:
            x = Conv2D(self.nClass, (3,3), kernel_initializer='he_normal', padding='same')(x)        
            e = Activation(self.activation)(x)
            if self.useScoreMapConnect==False or levelId<2:
                #e = Activation('softmax',name="softmax_{}_{}".format(levelId,blockTypeId))(e)        
                pass
            else:                
                eOld=self.E[(levelId-2)*3+blockTypeId]
                e=Add()([e,eOld])
            self.E[levelId*3+blockTypeId]=e
            b = Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, (self.height,self.width), align_corners=True))(e)            
            s = Activation('softmax',name="softmax_{}_{}".format(levelId,blockTypeId))(b)                    
            self.softmaxLayers.append(s)
        return output
        
    def predict(self,img):
        img=cv2.resize(img,(224,224))
        pred=self.model.predict(np.array([img]),verbose=0)
        pred=np.argmax(pred,axis=3)
        return pred[0] 
        
    def showResult(self,img,mask): 
        h,w,*_=img.shape
        seg=cv2.resize(colorlize(mask),(w,h)) 
        
        
        return seg*0.5+img*0.5
