import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

filePath_test = 'data/eval1.joblib'

size30 = 30
size50 = 50
size90 = 90

eval_data_90 = joblib.load(open(filePath_test, "rb"))[size90]
x_test_90 = eval_data_90["x_test"]
y_test_90 = eval_data_90["y_test"]

eval_data_50 = joblib.load(open(filePath_test, "rb"))[size50]
x_test_50 = eval_data_50["x_test"]
y_test_50 = eval_data_50["y_test"]

eval_data_30 = joblib.load(open(filePath_test, "rb"))[size30]
x_test_30 = eval_data_30["x_test"]
y_test_30 = eval_data_30["y_test"]


model30 = joblib.load('models/model30.joblib')
model50 = joblib.load('models/model50.joblib')
model90 = joblib.load('models/model90.joblib')


print(model30.score(x_test_30, y_test_30))
print(model50.score(x_test_50, y_test_50))
print(model90.score(x_test_90, y_test_90))

print("Confusion Matrix for 30: \n", confusion_matrix(y_test_30, model30.predict(x_test_30)))
print("Confusion Matrix for 50: \n", confusion_matrix(y_test_50, model50.predict(x_test_50)))
print("Confusion Matrix for 90: \n", confusion_matrix(y_test_90, model90.predict(x_test_90)))


print("Precision for 30: ", precision_score(y_test_30, model30.predict(x_test_30), average='macro'))
print("Precision for 50: ", precision_score(y_test_50, model50.predict(x_test_50), average='macro'))
print("Precision for 90: ", precision_score(y_test_90, model90.predict(x_test_90), average='macro'))

print("Recall for 30: ", recall_score(y_test_30, model30.predict(x_test_30), average='macro'))
print("Recall for 50: ", recall_score(y_test_50, model50.predict(x_test_50), average='macro'))
print("Recall for 90: ", recall_score(y_test_90, model90.predict(x_test_90), average='macro'))


print("F1 Score for 30: ", f1_score(y_test_30, model30.predict(x_test_30), average='macro'))
print("F1 Score for 50: ", f1_score(y_test_50, model50.predict(x_test_50), average='macro'))
print("F1 Score for 90: ", f1_score(y_test_90, model90.predict(x_test_90), average='macro'))