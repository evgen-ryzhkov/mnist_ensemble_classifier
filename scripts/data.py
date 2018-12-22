import mnist
import numpy as np

class MNISTData:
	# MNIST data set was separated on train (first 60 0000) and test (last 10 000) sets
	# mnist lib has special methods for getting train and test sets
	def get_train_set(self):
		X_train = mnist.train_images()
		y_train = mnist.train_labels()
		mixed_indexes = np.random.permutation(60000)
		X_train, y_train = X_train[mixed_indexes], y_train[mixed_indexes]
		X_train_prepared = self._prepare_data(X_train)
		return X_train_prepared, y_train

	def get_test_set(self):
		X_test = mnist.test_images()
		y_test = mnist.test_labels()
		X_test_prepared = self._prepare_data(X_test)
		return X_test_prepared, y_test

	def _prepare_data(self, not_prepared_data):
		# Normalizing the RGB codes by dividing it to the max RGB value.
		prepared_data = not_prepared_data / 255

		# used models needs 2 dimensions data
		prepared_data = prepared_data.reshape(prepared_data.shape[0], 784)

		# reduce RAM requirements
		prepared_data = prepared_data.astype(np.float32)

		return prepared_data

