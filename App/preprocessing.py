from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


class Preprocessing:
    @staticmethod
    def DropNull(df):
        df = df.dropna()
        return df

    @staticmethod
    def DropDuplicates(df):
        df = df.drop_duplicates()
        return df

    @staticmethod
    def Train_Test_Split(x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def Vectorization(x_train, x_test):
        vector = CountVectorizer()
        x_train_vec = vector.fit_transform(x_train)
        x_test_vec = vector.transform(x_test)
        return x_train_vec, x_test_vec

    @staticmethod
    def cm(y_test, y_prediction):
        confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
        return cm_display

    @staticmethod
    def ClusterVector(sentence):
        vector = TfidfVectorizer(stop_words='english')
        vectorized_documents = vector.fit_transform(sentence)
        return vectorized_documents

    @staticmethod
    def pca(data):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
        return reduced_data