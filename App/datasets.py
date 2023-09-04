import pandas as pd
from models import Models
from preprocessing import Preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler


class Datasets:
    #  -----------------------------------------------------------------------------------------------------------------
    #  --------------------------------------------- Email Dataset -----------------------------------------------------
    #  -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def email():
        data = pd.read_csv('App/spam_ham_dataset.csv')

        data.dropna(inplace=True)
        data.drop('Unnamed: 0', axis=1, inplace=True)

        x = data['text']
        y = data['label_num']

        # Split the data
        x_train, x_test, y_train, y_test = Preprocessing.Train_Test_Split(x, y)

        # Vectorization
        x_train_vec, x_test_vec = Preprocessing.Vectorization(x_train, x_test)

        # Naive base
        acc_email_naive, y_prediction_naive = Models.NaiveBayes(x_train_vec, x_test_vec, y_train, y_test)

        # Decision Tree
        acc_email_dt, y_prediction_dt = Models.DecisionTreeClass(x_train_vec, x_test_vec, y_train, y_test)

        # KNN
        acc_email_knn, y_prediction_knn = Models.knnClass(x_train_vec, x_test_vec, y_train, y_test, 3)

        # Confusion Matrix
        cm_naive = Preprocessing.cm(y_test, y_prediction_naive)
        cm_dt = Preprocessing.cm(y_test, y_prediction_dt)
        cm_knn = Preprocessing.cm(y_test, y_prediction_knn)

        return acc_email_naive, acc_email_dt, acc_email_knn, cm_naive, cm_dt, cm_knn

    #  -----------------------------------------------------------------------------------------------------------------
    #  --------------------------------------------- House Dataset -----------------------------------------------------
    #  -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def house():
        data = pd.read_csv('App/HousingData.csv')

        data['date'] = pd.to_datetime(data['date'])

        data['price'] = data['price'].astype('int64')
        data['bedrooms'] = data['bedrooms'].astype('int64')
        data['floors'] = data['floors'].astype('int64')
        data['street'] = data['street'].astype('string')
        data['city'] = data['city'].astype('string')
        data['statezip'] = data['statezip'].astype('string')
        data['country'] = data['country'].astype('string')

        data.insert(1, "year", data.date.dt.year)

        data['price'].replace(0, np.nan, inplace=True)
        data.dropna(inplace=True)

        data['age'] = data['year'] - data['yr_built']

        data = data[~(data['price'] > 0.3e7)]

        data['price'] = np.log1p(data['price'])

        X = pd.get_dummies(data.city, prefix='City')
        X_pca = Preprocessing.pca(X)
        data['city_pca1'] = X_pca[:, 0]
        data['city_pca2'] = X_pca[:, 1]

        df = data.drop(['date', 'street', 'statezip', 'country', 'year', 'city', 'age'], axis=1)

        x = df.drop("price", axis=1)
        y = pd.DataFrame(df["price"])

        x_train, x_test, y_train, y_test = Preprocessing.Train_Test_Split(x, y)

        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

        # Linear Regression
        acc_house_lr, mse_lr, rmse_lr, mae_lr, r2_lr = Models.LinearReg(x_train, x_test, y_train, y_test)

        # Decision Tree
        acc_house_dt, mse_dt, rmse_dt, mae_dt, r2_dt = Models.DecisionTreeRegression(x_train, x_test, y_train, y_test, 7)

        # KNN
        acc_house_knn, mse_knn, rmse_knn, mae_knn, r2_knn = Models.knnRegression(x_train, x_test, y_train, y_test, 20)

        # Evaluation Metrics
        em_lr = [mse_lr, rmse_lr, mae_lr, r2_lr]
        em_dt = [mse_dt, rmse_dt, mae_dt, r2_dt]
        em_knn = [mse_knn, rmse_knn, mae_knn, r2_knn]
        return acc_house_lr, acc_house_dt, acc_house_knn, em_lr, em_dt, em_knn

    #  -----------------------------------------------------------------------------------------------------------------
    #  ---------------------------------------------- Text Dataset -----------------------------------------------------
    #  -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def text():
        data = pd.read_json('App/sarcasm.json')

        sentence = data.headline

        y = data["is_sarcastic"]

        vectorized_documents = Preprocessing.ClusterVector(sentence)

        reduced_data = Preprocessing.pca(vectorized_documents.toarray())

        num_clusters = 2

        acc, predicted_labels = Models.K_mean(num_clusters, vectorized_documents, y, reduced_data, sentence)

        cm = Preprocessing.cm(y, predicted_labels)

        return acc, cm