"""
IMPORTANT: Keras has a bug: converting a masked element to nan. Run the file using: python coigan.py and if you face issues try re-running again.
"""
from __future__ import print_function, division
import scipy
import pandas
import keras.models
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pickle 

from usps_dataset import retrieve_uspsdataset
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# Create directory
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir('images/')
create_dir('datasets/')

class COIGAN():
	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.num_classes = 10
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 72

		optimizer = Adam(0.0002, 0.5)
		losses = ['binary_crossentropy', self.mutual_info_loss, 'binary_crossentropy', self.mutual_info_loss]

		# Build and the discriminator and recognition network
		self.d1, self.d2, self.auxilliary1, self.auxilliary2 = self.build_disk_and_q_net()
		
		self.d1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
		self.d2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
		self.g1, self.g2 = self.build_generators()
		self.g1.compile(loss='binary_crossentropy', optimizer=optimizer)
		self.g2.compile(loss='binary_crossentropy', optimizer=optimizer)

		# Build and compile the recognition network Q
		self.auxilliary1.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])
			
		self.auxilliary2.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # The generator takes noise as input and generated imgs
		gen_input1 = Input(shape=(self.latent_dim,))
		gen_input2 = Input(shape=(self.latent_dim,))

		img1 = self.g1(gen_input1)
		img2 = self.g2(gen_input2)

        # For the combined model we will only train the generators
		self.d1.trainable = False
		self.d2.trainable = False

        # The valid takes generated images as input and determines validity
		valid1 = self.d1(img1)
		valid2 = self.d2(img2)
		
		# The recognition network produces the label
		target_label1 = self.auxilliary1(img1)
		target_label2 = self.auxilliary2(img2)

        # The combined model  (stacked generators and discriminators) takes
        # noise as input => generates images => determines validity
		self.combined = Model([gen_input1, gen_input2], [valid1, valid2, target_label1, target_label2])
		self.combined.compile(loss=losses,
                                    optimizer=optimizer)

	#MAYBE TAKE THIS OUT?								
	def __call__(self):
		return self

	def build_generators(self):

		gen_input = Input(shape=(self.latent_dim,))

        # Shared weights between generators
		model = Sequential()
		model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
		model.add(Reshape((7, 7, 128)))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
		model.add(Activation("tanh"))

		latent = model(gen_input)

        # Generator 1
		g1 = Dense(1024)(latent)
		g1 = LeakyReLU(alpha=0.2)(g1)
		g1 = BatchNormalization(momentum=0.8)(g1)
		g1 = Dense(1, activation='tanh')(g1)
		img1 = Reshape(self.img_shape)(g1)

        # Generator 2
		g2 = Dense(1024)(latent)
		g2 = LeakyReLU(alpha=0.2)(g2)
		g2 = BatchNormalization(momentum=0.8)(g2)
		g2 = Dense(1, activation='tanh')(g2)
		img2 = Reshape(self.img_shape)(g2)

		model.summary()

		return Model(gen_input, img1), Model(gen_input, img2)

	def build_disk_and_q_net(self):

		img_shape = (self.img_rows, self.img_cols, self.channels)
		img1 = Input(shape=img_shape)
		img2 = Input(shape=img_shape)

        # Shared discriminator layers
		model = Sequential()
		model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
		model.add(ZeroPadding2D(padding=((0,1),(0,1))))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.25))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Flatten())

		img1_embedding = model(img1)
		img2_embedding = model(img2)

        # Discriminator 1
		validity1 = Dense(1, activation='sigmoid')(img1_embedding)
        # Discriminator 2
		validity2 = Dense(1, activation='sigmoid')(img2_embedding)
		
        # Recognition
		q_net1 = Dense(128, activation='relu')(img1_embedding)
		label1 = Dense(self.num_classes, activation='softmax')(q_net1)
		
		q_net2 = Dense(128, activation='relu')(img2_embedding)
		label2 = Dense(self.num_classes, activation='softmax')(q_net2)		

		return Model(img1, validity1), Model(img2, validity2), Model(img1, label1), Model(img2, label2)
		
	def mutual_info_loss(self, c, c_given_x):
		eps = 1e-8
		conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
		entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

		return conditional_entropy + entropy
		
	def sample_generator_input(self, batch_size):
        # Generator inputs
		sampled_noise = np.random.normal(0, 1, (batch_size, 62))
		sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
		sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

		return sampled_noise, sampled_labels

	def train(self, epochs, batch_size=128, save_interval=50):

		X_train_usps, y_train_usps, _, _ = retrieve_uspsdataset()
		(X_train_mnist, y_train_mnist), (_, _) = mnist.load_data()

		# Rescale -1 to 1 (usps)
		X_train_usps = (X_train_usps.astype(np.float32) - 127.5) / 127.5
		X_train_usps = np.expand_dims(X_train_usps, axis=3)
		y_train_usps = y_train_usps.reshape(-1, 1)

		X_train_mnist = (X_train_mnist.astype(np.float32) - 127.5) / 127.5
		X_train_mnist = np.expand_dims(X_train_mnist, axis=3)
		y_train_mnist = y_train_mnist.reshape(-1, 1)

		# Images in domain A and B (rotated)
		X1 = X_train_mnist #X_train[:int(X_train.shape[0]/2)]
		X2 = X_train_usps #X_train[int(X_train.shape[0]/2):]
		#X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

		half_batch = int(batch_size / 2)

		# Save parameters
		D1_loss_list = []
		D1_accuracy_list = []
		D2_loss_list = []
		D2_accuracy_list = []

		G_loss_param0 = []
		G_loss_param1 = []
		G_loss_param2 = []
		G_loss_param3 = []
		G_loss_param4 = []

		for epoch in range(epochs):

            # ----------------------
            #  Train Discriminators
            # ----------------------
			idx1 = np.random.randint(0, X1.shape[0], half_batch)
			idx2 = np.random.randint(0, X2.shape[0], half_batch)
			imgs1 = X1[idx1]
			imgs2 = X2[idx2]

			noise = np.random.normal(0, 1, (half_batch, 100))
			
            # Sample noise and categorical labels
			sampled_noise1, sampled_labels1 = self.sample_generator_input(half_batch)
			sampled_noise2, sampled_labels2 = self.sample_generator_input(half_batch)
			gen_input1 = np.concatenate((sampled_noise1, sampled_labels1), axis=1)			
			gen_input2 = np.concatenate((sampled_noise2, sampled_labels2), axis=1)
			
            # Generate a half batch of new images
			gen_imgs1 = self.g1.predict(gen_input1)
			gen_imgs2 = self.g2.predict(gen_input2)
			
			valid = np.ones((half_batch, 1))
			fake = np.zeros((half_batch, 1))			

            # Train the discriminators
			d1_loss_real = self.d1.train_on_batch(imgs1, valid)
			d2_loss_real = self.d2.train_on_batch(imgs2, valid)
			d1_loss_fake = self.d1.train_on_batch(gen_imgs1, fake)
			d2_loss_fake = self.d2.train_on_batch(gen_imgs2, fake)
			d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
			d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

            # -------------------------------
            #  Train Generators and Q-network
            # -------------------------------

            # Generator wants to fool the discriminator into believing that the generated
            # samples are real
			valid1 = np.ones((batch_size, 1))
			valid2 = np.ones((batch_size, 1))

            # Sample noise and categorical labels
			sampled_noise1, sampled_labels1 = self.sample_generator_input(batch_size)
			sampled_noise2, sampled_labels2 = self.sample_generator_input(batch_size)
			gen_input1 = np.concatenate((sampled_noise1, sampled_labels1), axis=1)
			gen_input2 = np.concatenate((sampled_noise2, sampled_labels2), axis=1)

            # Train the generators
			g_loss = self.combined.train_on_batch([gen_input1, gen_input2], [valid1, valid2, sampled_labels1, sampled_labels2])

			# Plot the progress
			print ("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G1 loss: %.2f] [G2 loss: %f]" % (epoch, d1_loss[0], 100*d1_loss[1], d2_loss[0], 100*d2_loss[1], g_loss[1], g_loss[2]))

			# Track loss values.
			D1_loss_list.append(d1_loss[0])
			D1_accuracy_list.append(100*d1_loss[1])
			D2_loss_list.append(d2_loss[0])
			D2_accuracy_list.append(100*d2_loss[1])
			G_loss_param0.append(g_loss[0])
			G_loss_param1.append(g_loss[1])
			G_loss_param2.append(g_loss[2])
			G_loss_param3.append(g_loss[3])
			G_loss_param4.append(g_loss[4])

            # If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.save_imgs(epoch)
		
		# Save metrics
		with open('losses.pkl', 'wb') as f: 
			pickle.dump([D1_loss_list, D1_accuracy_list, D2_loss_list, D2_accuracy_list, G_loss_param0, G_loss_param1, G_loss_param2, G_loss_param3, G_loss_param4], f)

	def save_imgs(self, epoch):
		r, c = 10, 10

		fig, axs = plt.subplots(r, c)
		for i in range(c):
			sampled_noise, _ = self.sample_generator_input(c)
			label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
			gen_input = np.concatenate((sampled_noise, label), axis=1)
			gen_imgs1 = self.g1.predict(gen_input)
			gen_imgs2 = self.g2.predict(gen_input)
			gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])
			for j in range(r):
				axs[j,i].imshow(gen_imgs[j,:,:,0], cmap='gray')
				axs[j,i].axis('off')
		fig.savefig("images/usps_%d.png" % epoch)
		plt.close()

	def save_model(self):

		def save(model, model_name):
			create_dir('saved_json/')
			create_dir('saved_hdf5')
			model_path = "saved_json/%s.json" % model_name
			weights_path = "saved_hdf5/%s_weights.hdf5" % model_name
			options = {"file_arch": model_path,
                        "file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])
			
		save(self.g1, "generator1")
		save(self.g2, "generator2")
		save(self.d1, "discriminator1")
		save(self.d2, "discriminator2")
		save(self.auxilliary1, "auxilliary1")
		save(self.auxilliary2, "auxilliary2")
		save(self.combined, "adversarial")	

		self.combined.save('whole_model.hdf5')
		
	def load_model(self):
		# Load the testing dataset
		(_, _), (x_test, y_test) = mnist.load_data()

		# Rescale -1 to 1
		x_test = (x_test.astype(np.float32) - 127.5) / 127.5
		x_test = np.expand_dims(x_test, axis=3)
		y_test = y_test.reshape(-1, 1)

		# Images in domain A and B (rotated)
		X1 = x_test[:int(x_test.shape[0]/2)]
		
		# Sample noise and categorical labels
		# Generator inputs
		num_classes = 10
		input_size = 5000
		sampled_noise = np.random.normal(0, 1, (input_size, 62))
		sampled_labels = np.random.randint(0, num_classes, input_size).reshape(-1, 1)
		sampled_labels = to_categorical(sampled_labels, num_classes=num_classes)
		gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

		model = model_from_json(open('saved_json/adversarial.json').read())
		model.load_weights('saved_hdf5/adversarial_weights.hdf5')

if __name__ == '__main__':
	model = COIGAN()
	model.train(epochs=30000, batch_size=64, save_interval=1)
	model.save_model()
	
	#model.load_model()