import pandas as pd
from sklearn import linear_model, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

HDD_CLEANED_PATH = 'datasets/cleaned/HDD-Cleaned.xlsx'
PKIOHD_CLEANED_PATH = 'datasets/cleaned/PKIOHD-Cleaned.xlsx'


class Model:

    @classmethod
    def separe_dataset(cls, features, label):
        train, test, train_labels, test_labels = train_test_split(
            features,
            label,
            test_size=0.30,
            random_state=42
        )
        return train, test, train_labels, test_labels


    @classmethod
    def evaluate_model(cls, test, model_predict):
        return metrics.classification_report(test, model_predict)


    @classmethod
    def gaussian_nb(cls, features, label):
        train, test, train_labels, test_labels = Model.separe_dataset(features, label)
        
        gnb = GaussianNB()
        gnb.fit(train, train_labels)
        predict = gnb.predict(test)

        # Avaliação do modelo
        metrics = Model.evaluate_model(test_labels, predict)
        print(f'GaussianNB: \n {metrics}')


    @classmethod
    def support_vector_machine(cls, features, label):
        train, test, train_labels, test_labels = Model.separe_dataset(features, label)

        svm_model = svm.SVC(
            kernel='linear',
            C=1.0,
        ).fit(train, train_labels)
        predict = svm_model.predict(test)

        # Avaliação do modelo
        metrics = Model.evaluate_model(test_labels, predict)
        print(f'Support Vector Machines: \n {metrics}')

    @classmethod
    def logistic_regression(cls, features, label):
        train, test, train_labels, test_labels = Model.separe_dataset(features, label)

        lr_model = linear_model.LogisticRegression(
            solver='liblinear',
            C=1.0
        )
        lr_model.fit(train, train_labels)
        predict = lr_model.predict(test)

        # Avaliação do modelo
        metrics = Model.evaluate_model(test_labels, predict)
        print(f'Logistic Regression: \n {metrics}')

    @classmethod
    def k_nearest_neighbors(cls, features, label):
        train, test, train_labels, test_labels = Model.separe_dataset(features, label)

        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(train, train_labels)
        predict = knn.predict(test)

        # Avaliação do modelo
        metrics = Model.evaluate_model(test_labels, predict)
        print(f'K-Nearest Neighbors: \n {metrics}')

if __name__ == '__main__':
    # Personal Indicators of Heart Disease
    pkiohd = pd.read_excel(PKIOHD_CLEANED_PATH)
    label = pkiohd['HeartDisease']
    features = pkiohd.drop('HeartDisease', axis=1)
    print('Personal Indicators of Heart Disease')
    Model.gaussian_nb(features, label)
    # Model.support_vector_machine(features, label)
    print('Personal Indicators of Heart Disease')
    Model.logistic_regression(features, label)
    print('Personal Indicators of Heart Disease')
    Model.k_nearest_neighbors(features, label)
    
    # Heart Disease Dataset
    hdd = pd.read_excel(HDD_CLEANED_PATH)
    label = hdd['target']
    features = hdd.drop('target', axis=1)
    print('Heart Disease Dataset')
    Model.gaussian_nb(features, label)
    print('Heart Disease Dataset')
    Model.support_vector_machine(features, label)
    print('Heart Disease Dataset')
    Model.logistic_regression(features, label)
    print('Heart Disease Dataset')
    Model.k_nearest_neighbors(features, label)