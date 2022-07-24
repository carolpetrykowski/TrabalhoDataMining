import pandas as pd
from sklearn import linear_model, metrics, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

HDD_CLEANED_PATH = 'datasets/cleaned/HDD-Cleaned.xlsx'
PKIOHD_CLEANED_PATH = 'datasets/cleaned/PKIOHD-Cleaned.xlsx'


class Model:

    @classmethod
    def split_dataset(cls, features, label):
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
    def create_cross_validation(cls, model, features, label):
        scores = cross_validate(
            model,
            features,
            label,
            cv=5,
            scoring=('accuracy', 'recall', 'f1', 'precision')
        )

        accuracy = scores.get('test_accuracy').mean()
        print(f'Acurácia: {accuracy:.2f}')

        recall = scores.get('test_recall').mean()
        print(f'Recall: {recall:.2f}')

        f1_score = scores.get('test_f1').mean()
        print(f'F1-Score: {f1_score:.2f}')

        precision = scores.get('test_precision').mean()
        print(f'Precisão: {precision:.2f}')

    @classmethod
    def gaussian_nb(cls, features, label):
        gnb = GaussianNB()

        # Validação cruzada
        print('Naïve Bayes: ')
        Model.create_cross_validation(gnb, features, label)

        return gnb

    @classmethod
    def support_vector_machine(cls, features, label):
        svm_model = svm.SVC(
            kernel='linear',
            C=1.0,
        )

        # Validação cruzada
        print('Support Vector Machine: ')
        Model.create_cross_validation(svm_model, features, label)

        return svm_model

    @classmethod
    def logistic_regression(cls, features, label):
        lr_model = linear_model.LogisticRegression(
            solver='liblinear',
            C=1.0
        )

        # Validação cruzada
        print('Regressão Logística: ')
        Model.create_cross_validation(lr_model, features, label)

        return lr_model

    @classmethod
    def k_nearest_neighbors(cls, features, label):
        knn = KNeighborsClassifier(n_neighbors=2)

        # Validação cruzada
        print('K-Nearest Neighbors: ')
        Model.create_cross_validation(knn, features, label)

        return knn

if __name__ == '__main__':
    # Personal Indicators of Heart Disease
    pkiohd = pd.read_excel(PKIOHD_CLEANED_PATH)
    label_pkiohd = pkiohd['HeartDisease']
    features_pkiohd = pkiohd.drop('HeartDisease', axis=1)

    print('--------------------------------------------------')
    print('Personal Key Indicators of Heart Disease - PKIOHD')
    Model.gaussian_nb(features_pkiohd, label_pkiohd)

    # print('--------------------------------------------------')
    # print('Personal Key Indicators of Heart Disease - PKIOHD')
    # Model.support_vector_machine(features_pkiohd, label_pkiohd)

    print('--------------------------------------------------')
    print('Personal Key Indicators of Heart Disease - PKIOHD')
    Model.logistic_regression(features_pkiohd, label_pkiohd)

    print('--------------------------------------------------')
    print('Personal Key Indicators of Heart Disease - PKIOHD')
    Model.k_nearest_neighbors(features_pkiohd, label_pkiohd)

    # Heart Disease Dataset
    hdd = pd.read_excel(HDD_CLEANED_PATH)
    label_hdd = hdd['target']
    features_hdd = hdd.drop('target', axis=1)

    print('-----------------------------')
    print('Heart Disease Dataset - HDD')
    Model.gaussian_nb(features_hdd, label_hdd)

    print('-----------------------------')
    print('Heart Disease Dataset - HDD')
    Model.support_vector_machine(features_hdd, label_hdd)

    print('-----------------------------')
    print('Heart Disease Dataset - HDD')
    Model.logistic_regression(features_hdd, label_hdd)

    print('-----------------------------')
    print('Heart Disease Dataset - HDD')
    knn = Model.k_nearest_neighbors(features_hdd, label_hdd)

    # Teste do melhor modelo com conjunto de teste
    train, test, train_labels, test_labels = Model.split_dataset(features_hdd, label_hdd)
    knn.fit(train, train_labels)
    predict = knn.predict(test)

    # Avaliação do modelo
    metrics = Model.evaluate_model(test_labels, predict)
    print(f'K-Nearest Neighbors: \n {metrics}')
