from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras import optimizers
import keras
import time
import scipy.misc
import os


model_weights_path = 'C:/Users/Ai/Desktop/Brawlhalla Supervised/Autoencoder_Final_weights.h5'
model_encoder_weights_path = 'C:/Users/Ai/Desktop/Brawlhalla Supervised/Autoencoder_Final_weights_ENCOER.h5'

class Autoencoder:
    def __init__(self, input_shape = (300,300,1), encoded_img = False, learning_rate = 0.01, load_Model = True):
        
        self.input_size = input_shape

        input_img = Input(shape=(input_shape))


        #https://arxiv.org/pdf/1806.00630.pdf
        
        input_img = Input(shape=(input_shape))



        
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        print ("shape of LAYER", K.int_shape(x))
        x = MaxPooling2D((2, 2), padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        x = MaxPooling2D((2, 2), padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        #x = MaxPooling2D((3, 3), padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        #x = MaxPooling2D((2, 2), padding='same')(x)
        self.encoded = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
        #self.encoded = MaxPooling2D((2, 2), padding='same')(x)


        #at this point the representation is ... 50*50*4
        x = Conv2D(4, (2, 2), activation='relu', padding='same')(self.encoded)
        print ("shape of LAYER", K.int_shape(x))
        #x = UpSampling2D((3, 3))(x)
        print ("shape of LAYER", K.int_shape(x))
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        x = UpSampling2D((2, 2))(x)
        print ("shape of LAYER", K.int_shape(x))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        x = UpSampling2D((2, 2))(x)
        print ("shape of LAYER", K.int_shape(x))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        print ("shape of LAYER", K.int_shape(x))
        #x = UpSampling2D((2, 2))(x)
        print ("shape of LAYER", K.int_shape(x))
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        print ("shape of encoded", K.int_shape(self.encoded))
        print ("shape of decoded", K.int_shape(self.decoded))
       
        #exit()



        self.autoencoder = Model(input_img, self.decoded)
        #self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.encoder_only = Model(input_img, self.encoded)

        

        self.autoencoder.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.adam(lr=learning_rate))
         



        if load_Model == True  and os.path.isfile(model_weights_path) == True:
            print("Loading models")
            self.load()











    def encode(self,x): # encodes img return dense representation
        predicted = self.encoder_only.predict(x)
        return predicted

    def run_whole(self,x): # decodes img 
        predicted = self.autoencoder.predict(x)
        return predicted

    def save(self): 
        #os.remove(model_weights_path)
        self.autoencoder.save_weights(model_weights_path)
        self.encoder_only.save_weights(model_encoder_weights_path)

        pass
    def load(self):
        self.autoencoder.load_weights(model_weights_path)
        self.encoder_only.load_weights(model_encoder_weights_path)

    def train(self, x_train, x_test,history, epochs=2, batch_size=20, shuffle=True ):
        from keras.callbacks import TensorBoard

        self.autoencoder.fit(x_train, x_train,
                
                epochs=epochs,#50
                batch_size=batch_size,
                shuffle=shuffle,
                validation_data=(x_test, x_test),
                callbacks=[history,TensorBoard(log_dir='C:/Users/Ai/Desktop/Brawlhalla Supervised/TENSORBOARD')])
        return history
        
    def IMG_RESULT(self, path, x_test, save = True, n = 10 ,figsize=(20, 4) ,size = (28, 28) ):#PLOTS IMAGES
        decoded_imgs = self.autoencoder.predict(x_test)
        
        #path = "C:/Users/Ai/Desktop/Brawlhalla Supervised/AUTOENCODERIMAGES/AUTOENCODER_"

        
        plt.figure()#figsize[0],figsize[1]
        for i in range(1,n+1):
            #save org
            timestr = time.strftime("%Y%m%d-%H%M%S")
            scipy.misc.imsave(path +'wholeimgORG' + timestr + '.jpg', x_test[i].reshape(self.input_size[0],self.input_size[1]))

            #save pred
            scipy.misc.imsave(path +'wholeimgpred' + timestr + '.jpg', decoded_imgs[i].reshape(self.input_size[0],self.input_size[1]))

            # display original
            ax = plt.subplot(2, n, i)
            plt.imshow(x_test[i].reshape(size[0],size[1]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i].reshape(size[0],size[1]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if save == False:
            plt.show()
            return
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(path + timestr+".jpeg")
        

#################TESTING#################
"""

autoencoderclass = Autoencoder((28, 28, 1),32,False)
autoencoder = autoencoderclass.autoencoder


from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format



from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=2,#50
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoderclass.ShowIMG()
"""