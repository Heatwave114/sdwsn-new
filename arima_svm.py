import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def svm_classify_regularity(entropy_csv_path):
    df = pd.read_csv(entropy_csv_path, index_col='node')
    
    X = df[ ['entropy'] ].values
    Y = df[['Target']].values

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X = StandardScaler().fit_transform(X)

    X_train , X_test, y_train, y_test = train_test_split(X,Y)

    clf = SVC(C=1.0, kernel='rbf').fit(X_train,y_train)

    # return clf
    print(clf)

svm_classify_regularity('C:/Users/sanis/Desktop/sdwsn-new/results/2021-09-15/02-33/remaining_energies/MLC/arima/aggregate/entropy-50.csv')
