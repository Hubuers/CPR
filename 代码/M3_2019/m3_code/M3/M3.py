import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
def load_data():
    url_M2 = 'E:/实验相关代码/m3代码/m3_code/data/M3/M3(80).xlsx'
    dfM2 = pd.read_excel(url_M2, engine='openpyxl')

    # 分别为输入（X）和输出（Y）变量
    X = dfM2.iloc[:, list(range(80))].astype(str).astype(float)
    y = dfM2['Label'].tolist()

    # 转化为numpy数组，导入keras模型
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_model():
    # create your model using this function
    # create model
    model = Sequential()
    model.add(Dense(500, input_shape=(80,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def draw_accuray_loss(history):
    # Plotting loss
    plt.plot(history.history['loss'])
    plt.title('Binary Cross Entropy Loss on Train data set')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # Plotting accuracy metric
    plt.plot(history.history['accuracy'])
    plt.title('Accuracy on the train data set')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

def train_and_evaluate__model(model, data_train, labels_train, data_test, labels_test):
    # Fit the model
    history = model.fit(data_train, labels_train, epochs=100, batch_size=128, verbose=0)
    # Draw accuracy and loss
    draw_accuray_loss(history)
    # fit and evaluate here.
    #loss, accuracy, f1_score, precision, recall = model.evaluate(data_test, labels_test, verbose=0)
    loss, accuracy = model.evaluate(data_test, labels_test, verbose=0)
    return accuracy

acc_scores = []
pre_scores = []
re_scores = []
f1_scores = []
auc_scores = []

if __name__ == "__main__":
    n_folds_list = [5]
    accuracy_list = []

    for n_folds in n_folds_list:
        data, labels = load_data()
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
        i = 0
        for train, test in skf.split(data, labels):
            print("Running Fold", i + 1, "/", n_folds)
            i += 1
            # model = None # Clearing the NN.
            model = create_model()
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(data[train])
            data[train] = scaler.transform(data[train])
            data[test] = scaler.transform(data[test])
            ## print
            print("Number transactions X_train dataset: ", data[train].shape)
            print("Number transactions y_train dataset: ", labels[train].shape)
            print("Number transactions X_test dataset: ", data[test].shape)
            print("Number transactions y_test dataset: ", labels[test].shape)
            print("Before OverSampling, counts of label '1':{}".format(sum(labels[train]==1)))
            print("Before OverSampling, counts of label '0':{}\n".format(sum(labels[train]==0)))
            sm = RandomOverSampler()
            X_train_res, y_train_res = sm.fit_resample(data[train], labels[train].ravel())
            # rus = RandomOverSampler(random_state=0)
            print('After OverSampling, the shape of train_X:{}'.format(X_train_res.shape))
            print('After OverSampling, the shape of train_y: {}\n'.format(y_train_res.shape))
            print("After OverSampling, counts of label '1':{}".format(sum(y_train_res == 1)))
            print("After OverSampling, counts of label '0':{}".format(sum(y_train_res == 0)))
            accuracy = train_and_evaluate__model(model, X_train_res, y_train_res, data[test], labels[test])

            pred_y = model.predict(data[test])
            acc = accuracy_score(labels[test], pred_y.round(), normalize=True)
            f1 = f1_score(labels[test], pred_y.round(), average="binary")
            precision = precision_score(labels[test], pred_y.round(), average="binary")
            recall = recall_score(labels[test], pred_y.round(), average="binary")
            try:
                auc = roc_auc_score(np.array(labels[test]), np.array(pred_y))
            except ValueError:
                pass
            print(auc)
            print("F1-score " + str(f1 * 100) + " Precision " + str(precision * 100) + " Recall " + str(recall * 100))
            print(confusion_matrix(labels[test], pred_y.round()))
            print(classification_report(labels[test], pred_y.round(), digits=4))
            acc_scores.append(acc * 100)
            pre_scores.append(precision * 100)
            re_scores.append(recall * 100)
            f1_scores.append(f1 * 100)
            auc_scores.append(auc)
            print("Accuracy:" + str(accuracy * 100))

    print("===========================================")
    print(numpy.mean(acc_scores))
    print(numpy.mean(pre_scores))
    print(numpy.mean(re_scores))
    print(numpy.mean(f1_scores))
    print(sum(auc_scores) / 5 * 100)
    print(auc_scores)

