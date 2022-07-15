import pandas as pd
import seaborn as sn 
import matplotlib.pyplot as plt
import numpy as np

HEART_ATTACK_ANALYSIS = 'datasets/HeartAttackAnalysisPredictionDataset.csv'
HEART_DISEASE_DATASET_PATH = 'datasets/HeartDiseaseDataset.csv'
HEART_FAILURE_PREDICTION_DATASET_PATH = 'datasets/HeartFailurePredictionDataset.csv'
PERSONAL_INDICATORS_HEART_DISEASE_PATH = 'datasets/PersonalKeyIndicatorsofHeartDisease.csv'

class PreparacaoDados:

    @classmethod
    def getDatasets(cls):

        # Heart Attack Analysis
        # columns_heart_attack_analysis = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
        # df_attack_analysis = pd.read_csv(HEART_ATTACK_ANALYSIS)
        # for feature in columns_heart_attack_analysis:
        #     PreparacaoDados.generate_box_plot(
        #         df_attack_analysis,
        #         'output',
        #         feature
        #     )
        
        # Heart Disease Dataset
        columns_heart_disease = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df_heart_disease = pd.read_csv(HEART_DISEASE_DATASET_PATH)
        for column in columns_heart_disease:
            PreparacaoDados.generate_box_plot(
                df_heart_disease,
                'target',
                column
            )

        # # Heart Failure Prediction Dataset
        # colums_heart_failure = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        # df_heart_failure = pd.read_csv(HEART_FAILURE_PREDICTION_DATASET_PATH)
        # for column in colums_heart_failure:
        #     PreparacaoDados.generate_scatter_plot(
        #         df_heart_failure,
        #         'HeartDisease',
        #         column
        #     )

        # Personal Key Indicators of Heart Disease
        colums_personal_indicators = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        df_personal_indicators_heart_disease = pd.read_csv(PERSONAL_INDICATORS_HEART_DISEASE_PATH)
        for column in colums_personal_indicators:
            PreparacaoDados.generate_box_plot(
                df_personal_indicators_heart_disease,
                'HeartDisease',
                column
            )

    # Variável alvo = Variável dependente = y
    # Feature = Variável independente = x
    @classmethod
    def generate_box_plot(cls, dataframe, target, feature):
        sn.boxplot(x=target, y=feature, data=dataframe, palette='hls')
        plt.show()

    @classmethod
    def generate_scatter_plot(cls, dataframe, target, feature):
        indices_sem_doenca = np.where(dataframe[target]==0)
        indices_sem_doenca = indices_sem_doenca[0].tolist()

        indices_com_doenca = np.where(dataframe[target]==1)
        indices_com_doenca = indices_com_doenca[0].tolist()    

        age_sem_doenca = dataframe[feature][indices_sem_doenca]
        age_com_doenca = dataframe[feature][indices_com_doenca]

        y_sem_doenca = dataframe[target][indices_sem_doenca]
        y_com_doenca = dataframe[target][indices_com_doenca]

        plt.scatter(age_sem_doenca, y_sem_doenca, color='blue')
        plt.scatter(age_com_doenca, y_com_doenca, color='red')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()    

    @classmethod
    def correlation(cls, dataframe):
        correlation = dataframe.corr()
        print(correlation)
        plot = (sn.heatmap(correlation, annot=True, fmt='.1f', linewidths=.6))
        plt.show()

    @classmethod
    def generate_pair_plot(cls, dataframe, target, columns_selected):
        # Gera um novo dataframe apenas com as colunas selecionadas
        novo_df = pd.DataFrame((dataframe[columns_selected].values), columns=columns_selected)
        
        # Pega os valores da variável alvo
        df_target = dataframe[target].values

        # Adiciona no novo dataframe a variável alvo
        novo_df[target] = pd.Series(df_target, dtype='category')

        # Gera scatterplots
        sn.pairplot(novo_df)
        plt.show()        

if __name__ == '__main__':
    PreparacaoDados.getDatasets()