# machine learning models
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Models:
    @staticmethod
    def SVM(x_train, y_train, x_test, y_test):
        svc = SVC()
        svc.fit(x_train, y_train)
        y_prediction = svc.predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        return accuracy, y_prediction

    @staticmethod
    def LinearReg(x_train, x_test, y_train, y_test):
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)

        accuracy = model.score(x_test, y_test)

        mse_lr = mean_squared_error(y_test, y_prediction)
        rmse_lr = mean_squared_error(y_test, y_prediction, squared=False)
        mae_lr = mean_absolute_error(y_test, y_prediction)
        r2_lr = r2_score(y_test, y_prediction)
        return accuracy, mse_lr, rmse_lr, mae_lr, r2_lr

    @staticmethod
    def LogisticReg(x_train, y_train, x_test, y_test):
        logistic_regression = LogisticRegression()
        logistic_regression.fit(x_train, y_train)
        y_prediction = logistic_regression.predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        return accuracy, y_prediction

    @staticmethod
    def knnClass(x_train, x_test, y_train, y_test, n_neighbors):
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model.fit(x_train, y_train)
        y_prediction = knn_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        return accuracy, y_prediction

    @staticmethod
    def DecisionTreeClass(x_train, x_test, y_train, y_test):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)
        y_prediction = decision_tree.predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        return accuracy, y_prediction

    @staticmethod
    def knnRegression(x_train, x_test, y_train, y_test, n_neighbors):
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
        knn_model.fit(x_train, y_train)
        y_prediction = knn_model.predict(x_test)

        accuracy = knn_model.score(x_test, y_test)

        mse_knn = mean_squared_error(y_test, y_prediction)
        rmse_knn = mean_squared_error(y_test, y_prediction, squared=False)
        mae_knn = mean_absolute_error(y_test, y_prediction)
        r2_knn = r2_score(y_test, y_prediction)

        return accuracy, mse_knn, rmse_knn, mae_knn, r2_knn

    @staticmethod
    def DecisionTreeRegression(x_train, x_test, y_train, y_test, max_depth):
        decision_tree = DecisionTreeRegressor(max_depth=max_depth)
        decision_tree.fit(x_train, y_train)
        y_prediction = decision_tree.predict(x_test)

        accuracy = decision_tree.score(x_test, y_test)

        mse_dt = mean_squared_error(y_test, y_prediction)
        rmse_dt = mean_squared_error(y_test, y_prediction, squared=False)
        mae_dt = mean_absolute_error(y_test, y_prediction)
        r2_dt = r2_score(y_test, y_prediction)
        return accuracy, mse_dt, rmse_dt, mae_dt, r2_dt

    @staticmethod
    def NaiveBayes(x_train, x_test, y_train, y_test):
        naive_model = MultinomialNB()
        naive_model.fit(x_train, y_train)
        y_prediction = naive_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_prediction)
        return accuracy, y_prediction

    @staticmethod
    def K_mean(num_clusters, data, y, reduced_data, sentence):
        k_means = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500, random_state=42)
        k_means.fit(data)

        # results = pd.DataFrame()
        # results['document'] = sentence
        # results['cluster'] = k_means.labels_
        #
        # colors = ['red', 'green', 'blue', 'black']
        # cluster = ['Not Sarcastic', 'Sarcastic']
        # for i in range(num_clusters):
        #     plt.scatter(reduced_data[k_means.labels_ == i, 0],
        #                 reduced_data[k_means.labels_ == i, 1],
        #                 s=10, color=colors[i],
        #                 label=f' {cluster[i]}')
        # plt.legend()
        # plt.show()

        cluster_labels = k_means.labels_

        cluster_labels_mapping = {}
        for cluster in set(cluster_labels):
            mask = (cluster_labels == cluster)
            assigned_labels = y[mask]
            unique_labels, counts = np.unique(assigned_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            cluster_labels_mapping[cluster] = majority_label

        predicted_labels = [cluster_labels_mapping[label] for label in cluster_labels]

        accuracy = accuracy_score(y, predicted_labels)

        return accuracy, predicted_labels