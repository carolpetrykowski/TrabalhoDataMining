import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn 

HEART_DISEASE_DATASET_PATH = 'datasets/HeartDiseaseDataset.csv'
PERSONAL_INDICATORS_HEART_DISEASE_PATH = 'datasets/PersonalKeyIndicatorsofHeartDisease.csv'


class Analise:

    @classmethod
    def analisys(cls):
        # Heart Disease Dataset
        columns_heart_disease = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df_heart_disease = pd.read_csv(HEART_DISEASE_DATASET_PATH)
        for column in columns_heart_disease:
            Analise.generate_box_plot(
                df_heart_disease,
                'target',
                column
            )

        # Personal Key Indicators of Heart Disease
        colums_personal_indicators = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        df_personal_indicators_heart_disease = pd.read_csv(PERSONAL_INDICATORS_HEART_DISEASE_PATH)
        for column in colums_personal_indicators:
            Analise.generate_box_plot(
                df_personal_indicators_heart_disease,
                'HeartDisease',
                column
            )

    @classmethod
    def generate_box_plot(cls, dataframe, target, feature):
        sn.boxplot(x=target, y=feature, data=dataframe, palette='hls')
        plt.show()

if __name__ == '__main__':
    Analise.analisys() # Boxplot