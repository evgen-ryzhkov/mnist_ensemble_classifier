from scripts.data import MNISTData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data and familiarity with it

data_obj = MNISTData()
X_train, y_train = data_obj.get_train_set()


#  --------------
# Step 2
# training models
clf_o = DigitClassifier()
final_model = clf_o.train_models(X_train, y_train)


#  --------------
# Step 3
# testing model
X_test, y_test = data_obj.get_test_set()
y_pred = final_model.predict(X_test)
clf_o._calculate_model_metrics(y_test, y_pred, 'Final model with test data set:')
