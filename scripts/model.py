from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

class DigitClassifier:

	def train_models(self, X_train, y_train):
		rnd_forest_clf = self._train_rnd_forest_clf(X_train, y_train)
		print('\n--------------------------------------------------')
		extra_trees_clf = self._train_extra_trees_clf(X_train, y_train)
		print('\n--------------------------------------------------')
		svm_clf = self._train_svm_clf(X_train, y_train)
		print('\n--------------------------------------------------')
		final_model = self._create_ensemble(X_train, y_train, rnd_forest_clf, extra_trees_clf, svm_clf)
		return final_model

	def _train_rnd_forest_clf(self, X_train, y_train):
		print('Train RandomForestClassifier was started...')
		clf = RandomForestClassifier()
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='RandomForestClassifier')
		return clf

	def _train_extra_trees_clf(self, X_train, y_train):
		print('Train ExtraTreesClassifier was started...')
		clf = ExtraTreesClassifier()
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='ExtraTreesClassifier')
		return clf

	def _train_svm_clf(self, X_train, y_train):
		print('Train SVM classifier was started...')
		clf = SVC(probability=True)
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='SVM classifier')
		return clf

	def _create_ensemble(self, X_train, y_train, clf_1, clf_2, clf_3):
		print('Train Ensemble was started...')
		voting_clf = VotingClassifier(
			estimators=[('rnd_forest', clf_1), ('extra_trees', clf_2), ('svm', clf_3)],
			voting='soft'
		)
		voting_clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(voting_clf, X_train, y_train, cv=3)
		self._calculate_model_metrics(y_train, y_train_pred, model_name='Ensemble classifier')
		return voting_clf

	@staticmethod
	def _calculate_model_metrics(y_train, y_pred, model_name):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			y_train, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(model_name, '\n')
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())




