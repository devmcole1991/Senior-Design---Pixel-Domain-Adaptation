from __future__ import print_function, division
import scipy
import pandas
import keras.models
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pickle 

from keras_contrib.layers.normalization import InstanceNormalization
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda, Add
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
from data_loader import DataLoader

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
		self.channels = 3
		self.num_classes = 10
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		self.latent_dim = 72
		
		# Configure MNIST and MNIST-M data loader
		self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

		# Loss weights
		lambda_adv = 10
		lambda_clf = 1

		# Calculate output shape of D (PatchGAN)
		#patch = int(self.img_rows/ 2**4)
		self.disc_patch = (2, 2, 1)

		# Number of residual blocks in the generator
		self.residual_blocks = 6

		optimizer = Adam(0.0002, 0.5)
		
		# Number of filters in first layer of discriminator and classifier
		self.df = 64
		self.cf = 64

		# Build and the discriminator and recognition network
		#self.d1, self.d2, self.auxilliary1, self.auxilliary2 = self.build_disk_and_q_net()
		self.d1, self.d2 = self.build_discriminators()
		self.d1.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
		self.d2.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
		self.g1, self.g2 = self.build_generators()
		self.g1.compile(loss='binary_crossentropy', optimizer=optimizer)
		self.g2.compile(loss='binary_crossentropy', optimizer=optimizer)

		# Build and compile the task (classification) network
		self.clf1, self.clf2 = self.build_classifiers()
		self.clf1.compile(loss='binary_crossentropy', optimizer=optimizer)
		self.clf2.compile(loss='binary_crossentropy', optimizer=optimizer)

		# Input images from both domains
		img_A = Input(shape=self.img_shape)
		img_B = Input(shape=self.img_shape)

		# Translate images from domain A to domain B
		fake_1 = self.g1(img_A)
		fake_2 = self.g2(img_B)

		# Classify the translate images
		class_pred1 = self.clf1(fake_1)
		class_pred2 = self.clf2(fake_2)

        # For the combined model we will only train the generators
		self.d1.trainable = False
		self.d2.trainable = False

        # Discriminator determines validity of translated images
		valid1 = self.d1(fake_1)
		valid2 = self.d2(fake_2)
		
		self.combined = Model([img_A, img_B], [valid1, class_pred1, valid2, class_pred2])
		self.combined.compile(loss=['mse', 'categorical_crossentropy', 'mse', 'categorical_crossentropy'],
									loss_weights=[lambda_adv, lambda_clf, lambda_adv, lambda_clf],
									optimizer=optimizer,
									metrics=['accuracy'])

	def __call__(self):
		return self

	def build_generators(self):

		"""Resnet Generator"""
		"""
		def residual_block(model):
			#Residual block described in paper
			model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
			model.add(BatchNormalization(momentum=0.8))
			model.add(Activation('relu'))
			model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))
			model.add(BatchNormalization(momentum=0.8))
			model.add(Add(), output_dim=[d, layer_input])
            
			return model

		# Image input
		model = Sequential()
		img = Input(shape=self.img_shape)

		model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_dim = img))

		# Propogate signal through residual blocks
		r = residual_block(model, l1)
		for _ in range(self.residual_blocks - 1):
			r = residual_block(r)

		output_img1 = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(r)
		output_img2 = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(r)

		return Model(img, output_img1), Model(img, output_img2)
		"""

		def residual_block(layer_input):
			#Residual block described in paper
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(layer_input)
			d = BatchNormalization(momentum=0.8)(d)
			d = Activation('relu')(d)
			d = Conv2D(64, kernel_size=3, strides=1, padding='same')(d)
			d = BatchNormalization(momentum=0.8)(d)
			d = Add()([d, layer_input])
			return d

		# Image input
		img = Input(shape=self.img_shape)

		l1 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(img)

		# Propogate signal through residual blocks
		r = residual_block(l1)
		for _ in range(self.residual_blocks - 1):
			r = residual_block(r)

		output_img = Conv2D(self.channels, kernel_size=3, padding='same', activation='tanh')(r)

		return Model(img, output_img), Model(img, output_img)


	def build_discriminators(self):
		"""
		def d_layer(model, layer_input, filters, img1, f_size=4, normalization=True):
			#Discriminator layer
			model.add(Conv2D(filters, kernel_size=f_size, strides=2, padding='same', input_dim=img1))
			model.add(LeakyReLU(alpha=0.2))
			if normalization:
				model.add(InstanceNormalization())
			return model

		img1 = Input(shape=self.img_shape)
		img2 = Input(shape=self.img_shape)
		model = Sequential()

		d1_1 = d_layer(model, self.img_shape, img1, self.df, normalization=False)
		d1_2 = d_layer(model, d1, self.df*2)
		d1_3 = d_layer(model, d2, self.df*4)
		d1_4 = d_layer(model, d3, self.df*8)

		img1_embedding = model(img1)
		img2_embedding = model(img2)

		validity1 = Conv2D(1, kernel_size=4, strides=1, padding='same')(img1_embedding)
		validity2 = Conv2D(1, kernel_size=4, strides=1, padding='same')(img2_embedding)

		return Model(img1, validity1), Model(img2, validity2)
		"""
		def d_layer(layer_input, filters, f_size=4, normalization=True):
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape)

		d1 = d_layer(img, self.df, normalization=False)
		d2 = d_layer(d1, self.df*2)
		d3 = d_layer(d2, self.df*4)
		d4 = d_layer(d3, self.df*8)

		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

		return Model(img, validity), Model(img, validity)

	def build_classifiers(self):
		
		"""
		def clf_layer(layer_input, filters, f_size=4, normalization=True):
			#Classifier layer
			model.add(Conv2D(filters, kernel_size=f_size, strides=2, padding='same', input_dim='layer_input'))
			model.add(LeakyReLU(alpha=0.2))
			if normalization:
				model.add(InstanceNormalization())
			return model

		model = Sequential()
		img = Input(shape=self.img_shape)

		c1 = clf_layer(model, img, self.cf, normalization=False)
		c2 = clf_layer(model, c1, self.cf*2)
		c3 = clf_layer(model, c2, self.cf*4)
		c4 = clf_layer(model, c3, self.cf*8)
		c5 = clf_layer(model, c4, self.cf*8)

		class_pred = Dense(self.num_classes, activation='softmax')(Flatten()(c5))

		return Model(img, class_pred)
		"""

		def clf_layer(layer_input, filters, f_size=4, normalization=True):
			"""Classifier layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if normalization:
				d = InstanceNormalization()(d)
			return d

		img = Input(shape=self.img_shape)

		c1 = clf_layer(img, self.cf, normalization=False)
		c2 = clf_layer(c1, self.cf*2)
		c3 = clf_layer(c2, self.cf*4)
		c4 = clf_layer(c3, self.cf*8)
		c5 = clf_layer(c4, self.cf*8)

		class_pred = Dense(self.num_classes, activation='softmax')(Flatten()(c5))

		return Model(img, class_pred), Model(img, class_pred)

	def train(self, epochs, batch_size=128, save_interval=50):

		batch_size = 64
		half_batch = int(batch_size / 2)

		test_accs = []

		# Save parameters
		D1_loss_list = []
		D1_accuracy_list = []
		D2_loss_list = []
		D2_accuracy_list = []

		G1_loss_param1 = []
		Classifier_loss1 = []
		G1_accuracy_list = []
		G1_test_accuracy_list1 = []
		G1_test_accuracy_list2 = []
		
		G2_loss_param1 = []
		G2_accuracy_list = []
		G2_test_accuracy_list1 = []
		G2_test_accuracy_list2 = []

		for epoch in range(epochs):

			# ----------------------
			#  Train Discriminators
			# ----------------------

			# Select a random half batch of images
			imgs1, _ = self.data_loader.load_data(domain="A", batch_size=half_batch)
			imgs2, _ = self.data_loader.load_data(domain="B", batch_size=half_batch)

			# Translate images from domain A to domain B
			fake_1 = self.g1.predict(imgs1)
			fake_2 = self.g2.predict(imgs2)
			
			valid1 = np.ones((half_batch,) + self.disc_patch)
			valid2 = np.ones((half_batch,) + self.disc_patch)
			fake1 = np.zeros((half_batch,) + self.disc_patch)		
			fake2 = np.zeros((half_batch,) + self.disc_patch)
			
			#print(valid1.shape)
			#print(fake1.shape)
			# Train the discriminators
			d1_loss_real = self.d1.train_on_batch(imgs1, valid1)
			#sys.exit()
			d2_loss_real = self.d2.train_on_batch(imgs2, valid2)
			d1_loss_fake = self.d1.train_on_batch(fake_1, fake1)
			d2_loss_fake = self.d2.train_on_batch(fake_2, fake2)
			d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
			d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

			# -------------------------------
			#  Train Generators and Q-network
			# -------------------------------
			
			# Sample a batch of images from both domains
			imgs_1, labels_1 = self.data_loader.load_data(domain="A", batch_size=half_batch)
			imgs_2, labels_2 = self.data_loader.load_data(domain="B", batch_size=half_batch)

			# One-hot encoding of labels
			labels_1 = to_categorical(labels_1, num_classes=self.num_classes)
			labels_2 = to_categorical(labels_2, num_classes=self.num_classes)

			# The generators want the discriminators to label the translated images as real
			valid1 = np.ones((half_batch,) + self.disc_patch)
			valid2 = np.ones((half_batch,) + self.disc_patch)

			# Train the generator and classifier
			g_loss = self.combined.train_on_batch([imgs_1, imgs2], [valid1, labels_1, valid2, labels_2])
			
			#-----------------------
			# Evaluation (domain B)
			#-----------------------

			pred_2 = self.clf2.predict(imgs_2)
			test_acc = np.mean(np.argmax(pred_2, axis=1) == labels_2)
			
			# Add accuracy to list of last 100 accuracy measurements
			test_accs.append(test_acc)
			if len(test_accs) > 100:
				test_accs.pop(0)

			# Plot the progress
			print ( "%d : [D1 loss: %.5f, acc: %3d%%], [D2 loss: %.5f, acc: %3d%%], [G1 loss: %.5f], [clf loss: %.5f, acc: %3d%%, test_acc: %3d%% (%3d%%)]" % \
											(epoch, 
											d1_loss[0], 100*float(d1_loss[1]),
											d2_loss[0], 100*float(d2_loss[1]),
											g_loss[1], g_loss[2], 100*float(g_loss[-1]),
											100*float(test_acc), 100*float(np.mean(test_accs))))

			# Track loss values.
			D1_loss_list.append(d1_loss[0])
			D1_accuracy_list.append(100*d1_loss[1])
			D2_loss_list.append(d2_loss[0])
			D2_accuracy_list.append(100*d2_loss[1])

			G1_loss_param1.append(g_loss[1])
			Classifier_loss1.append(g_loss[2])
			G1_accuracy_list.append(100*float(g_loss[-1]))
			G1_test_accuracy_list1.append(100*float(test_acc))
			G1_test_accuracy_list2.append(100*float(np.mean(test_accs)))

			"""
			G2_loss_param1.append(g2_loss[1])
			Classifier_loss2.append(g2_loss[2])
			G2_accuracy_list.append(100*float(g2_loss[-1]))
			G2_test_accuracy_list1.append(100*float(test_acc))
			G2_test_accuracy_list2.append(100*float(np.mean(test_accs)))
			"""
			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
				self.sample_images(epoch)

		# Save metrics
		with open('losses.pkl', 'wb') as f: 
			pickle.dump([D1_loss_list, D1_accuracy_list, D2_loss_list, D2_accuracy_list,
						G1_loss_param1, Classifier_loss1, G1_accuracy_list, G1_test_accuracy_list1, G1_test_accuracy_list2], f)

	def sample_images(self, epoch):
		r, c = 2, 5

		imgs_A, _ = self.data_loader.load_data(domain="A", batch_size=5)

		# Translate images to the other domain
		fake_B = self.g2.predict(imgs_A)

		gen_imgs = np.concatenate([imgs_A, fake_B])

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		#titles = ['Original', 'Translated']
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt])
				#axs[i, j].set_title(titles[i])
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/%d.png" % (epoch))
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