import pandas as pd

HEART_DISEASE_DATASET_PATH = 'datasets/HeartDiseaseDataset.csv'
PERSONAL_INDICATORS_HEART_DISEASE_PATH = 'datasets/PersonalKeyIndicatorsofHeartDisease.csv'

class Preparation:

    @classmethod
    def prepare(cls):
        df_pihd = pd.read_csv(PERSONAL_INDICATORS_HEART_DISEASE_PATH)
        
        df_pihd_cleaned = df_pihd

        df_pihd_cleaned = df_pihd_cleaned.drop(
            columns=[
                'Race',
                'AgeCategory',
                'PhysicalHealth',
                'MentalHealth',
                'PhysicalActivity',
                'GenHealth',
                'SleepTime',
                'Asthma',
            ]
        )

        df_pihd_cleaned = df_pihd_cleaned.replace(
            {
                'No': 0,
                'Yes': 1,
                'No, borderline diabetes': 0,
                'Yes (during pregnancy)': 1,
                'Female': 1,
                'Male': 0,
            }
        )

        df_pihd_cleaned.to_excel('datasets/cleaned/PKIOHD-Cleaned.xlsx', index=False)

if __name__ == '__main__':
    Preparation.prepare() # Preparação dos dados