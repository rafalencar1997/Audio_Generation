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

        # Generate Models
        self.G = generator(NOISE_DIM, AUDIO_SHAPE)
        self.D = discriminator(AUDIO_SHAPE)
        self.stacked_G_D = stacked_G_D(self.G, self.D)

        # Compile Models
        self.Optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
        self.G.compile(loss='binary_crossentropy', optimizer=self.Optimizer, metrics=['accuracy'])
        self.D.compile(loss='binary_crossentropy', optimizer=self.Optimizer, metrics=['accuracy'])
        self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.Optimizer,  metrics=['accuracy'])

        self.X_train = normalization(load_train_data(AUDIO_SHAPE, label))

        self.D_loss_hist = []
        self.G_loss_hist = []

    # Train GAN
    def train(self, epochs=20, batch=32, save_interval=2):
        for cnt in range(epochs):
          
            # Train discriminator
            halfBatch = int(batch/2)
            random_index = np.random.randint(0, len(self.X_train) - halfBatch)
            legit_audios = self.X_train[random_index: int(random_index + halfBatch)]
            gen_noise = np.random.normal(0, 1, (halfBatch, NOISE_DIM))
            syntetic_audios = self.G.predict(gen_noise)
            x_combined_batch = np.concatenate((legit_audios, syntetic_audios))
            y_combined_batch = np.concatenate((np.ones((halfBatch, 1)), np.zeros((halfBatch, 1))))

            # Update stacked discriminator Weights
            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

            self.stacked_G_D.layers[1].set_weights(self.D.get_weights())
            stacked_D_Weights = self.stacked_G_D.layers[1].get_weights()
            D_Weights = self.D.get_weights()

            d_loss_mean = np.mean(d_loss)
            self.D_loss_hist.append(d_loss_mean)

            # Train stacked generator
            noise = np.random.normal(0, 1, (batch, NOISE_DIM))
            y_mislabled = np.ones((batch, 1))
            g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabled)

            # Update generator Weights
            self.G.set_weights(self.stacked_G_D.layers[0].get_weights())

            g_loss_mean = np.mean(g_loss)
            self.G_loss_hist.append(g_loss_mean)
            if cnt % int(save_interval/2) == 0:
                print("epoch: %d" % (cnt))
                print("Discriminator_loss: %f, Generator_loss: %f" % (d_loss_mean, g_loss_mean))
            if cnt % save_interval == 0:
                self.show_samples(4)
            
    def show_samples(self, samples = 4):
        fig     = plt.figure(figsize=(1, samples))
        samplePlot = []
        noise = np.random.normal(0, 1, (samples,NOISE_DIM))
        audios = self.G.predict(noise)
        for i, audio in enumerate(audios):
            IPython.display.display(ipd.Audio(data=audio, rate=SAMPLE_RATE))
            samplePlot.append(fig.add_subplot(1, samples, i+1))
            samplePlot[i].plot(audio.flatten(), '.', )
        plt.gcf().set_size_inches(25, 5)
        plt.subplots_adjust(wspace=0.3,hspace=0.3)
        plt.show()