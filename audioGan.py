import numpy as numpy
import matplotlib
import matplotlib.pyplot as plt
import IPython
import IPython.display as ipd 

from keras.optimizers import Adam
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from ganModels import *
from ganSetup import *

class AudioGAN:
    def __init__(self, label="Saxophone"):

        # Generate models
        self.enc = encoder(AUDIO_SHAPE, ENCODE_SIZE)
        self.gen = generator(NOISE_DIM, AUDIO_SHAPE)
        self.dis = discriminator(AUDIO_SHAPE)
        self.stackGenDis = stacked_G_D(self.gen, self.dis)
        self.autoencoder = autoEncoder(self.enc, self.gen)  
        
        # Compile models
        self.opt = Adam(lr = 0.0002, beta_1 = 0.9)
        self.gen.compile(loss = 'binary_crossentropy', optimizer = self.opt, metrics = ['accuracy'])
        self.dis.compile(loss = 'binary_crossentropy', optimizer = self.opt, metrics = ['accuracy'])
        self.autoencoder.compile(loss = 'mse',         optimizer = self.opt, metrics = ['accuracy'])
        self.stackGenDis.compile(loss = 'binary_crossentropy', optimizer = self.opt,  metrics = ['accuracy'])

        # Set training data
        self.trainData = normalization(load_train_data(AUDIO_SHAPE, label))

        self.disLossHist = []
        self.genLossHist = []

    # Train GAN
    def train_gan(self, epochs = 20, batch = 32, save_interval = 2):
        for cnt in range(epochs):
          
            # Train discriminator
            halfBatch = int(batch/2)
            random_index = np.random.randint(0, len(self.trainData) - halfBatch)
            legit_audios = self.trainData[random_index: int(random_index + halfBatch)]
            gen_noise = np.random.normal(0, 1, (halfBatch, NOISE_DIM))
            syntetic_audios = self.gen.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_audios, syntetic_audios))
            y_combined_batch = np.concatenate((np.ones((halfBatch, 1)), np.zeros((halfBatch, 1))))
            d_loss = self.dis.train_on_batch(x_combined_batch, y_combined_batch)
            
            # Update stacked discriminator weights
            self.stackGenDis.layers[1].set_weights(self.dis.get_weights())
    
            # Include discriminator loss
            d_loss_mean = np.mean(d_loss)
            self.disLossHist.append(d_loss_mean)

            # Train stacked generator
            noise = np.random.normal(0, 1, (batch, NOISE_DIM))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stackGenDis.train_on_batch(noise, y_mislabled)

            # Update generator Weights
            self.gen.set_weights(self.stackGenDis.layers[0].get_weights())
            
            # Include generator loss
            g_loss_mean = np.mean(g_loss)
            self.genLossHist.append(g_loss_mean)
            
            if cnt % int(save_interval/2) == 0:
                print("epoch: %d" % (cnt))
                print("Discriminator_loss: %f, Generator_loss: %f" % (d_loss_mean, g_loss_mean))
            if cnt % save_interval == 0:
                self.show_gen_samples(4)
     
    # Plot a number of generated samples
    def show_gen_samples(self, samples = 4):
        samplePlot = []
        fig        = plt.figure(figsize = (1, samples))
        noise      = np.random.normal(0, 1, (samples,NOISE_DIM))
        audios     = self.gen.predict(noise)        
        for i, audio in enumerate(audios):
            IPython.display.display(ipd.Audio(data = audio, rate = SAMPLE_RATE))
            samplePlot.append(fig.add_subplot(1, samples, i+1))
            samplePlot[i].plot(audio.flatten(), '.', )
        plt.gcf().set_size_inches(25, 5)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.show()
       
    # Train autoencoder
    def train_autoencoder(self, epochs = 20,  save_internal = 2, batch = 32):
        for cnt in range(epochs):
            random_index = np.random.randint(0, len(self.trainData) - batch)
            legit_audios = self.trainData[random_index : int(random_index + batch)]
            loss = self.autoencoder.train_on_batch(legit_audios, legit_audios)
            if cnt % save_internal == 0 : 
                print("Epoch: ", cnt, ", Loss: ", loss)