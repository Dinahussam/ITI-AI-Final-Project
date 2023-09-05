import streamlit as st
from datasets import Datasets
import matplotlib.pyplot as plt
from PIL import Image
import base64

st.set_page_config(
    page_title="AI Models",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------- Styling ------------------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


colLogo, colSpace = st.columns([0.5, 2])


def render_svg(svg):
    """Renders the given svg string."""
    svg=open(svg).read()
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img class="logo" src="data:image/svg+xml;base64,%s"/>' % b64
    colLogo.write(html, unsafe_allow_html=True)


render_svg("./Images/App images/logo.svg")

with open("App/style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

# -------------------------------------------------- Main --------------------------------------------------------------
colSpace.write("# Models")
col1, col2 = st.columns([3, 2])
col1.write("## Choose The Dataset ")
Mode = col1.selectbox(label="", options=[
                    'Email Spam Detection Dataset', 'Document Classification Dataset',
                    'Platform Price Prediction Dataset'])

chBxCol1, chBxCol2 = st.columns([1, 3])

# -------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Emails Dataset ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
if Mode == 'Email Spam Detection Dataset':
    acc_email_naive, acc_email_dt, acc_email_knn, cm_naive, cm_dt, cm_knn = Datasets.email()
    acc = [acc_email_naive, acc_email_dt, acc_email_knn]
    labels = ['Naive Bias', 'Decision Tree', 'KNN']
    cm = [cm_naive, cm_dt, cm_knn]
    agree1 = col1.checkbox('Naive Bayes')
    agree2 = col1.checkbox('Decision Tree')
    agree3 = col1.checkbox('KNN')
    col1.write("")
    conMx = chBxCol1.checkbox('Confusion Matrix')
    Accuracy = chBxCol2.checkbox('Accuracy')

    cmCol1, cmCol2, cmCol3 = st.columns([2, 2, 2])

    # Accuracy Conditions
    if (agree1 == Accuracy == True) and (agree2 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[0], acc[0])
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree2 == Accuracy == True) and (agree1 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[1], acc[1])
        plt.text(0, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree3 == Accuracy == True) and (agree1 == agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[2], acc[2])
        plt.text(0, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree1 == agree2 == Accuracy == True) and (agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[0:2], acc[0:2])
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree2 == agree3 == Accuracy == True) and (agree1 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[1:], acc[1:])
        plt.text(0, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.text(1, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree1 == agree3 == Accuracy == True) and (agree2 == False):
        labels2 = ['Naive Bias', 'KNN']
        acc2 = [acc_email_naive, acc_email_knn]
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels2, acc2)
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif agree1 and agree2 and agree3 and Accuracy:
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels, acc)
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.text(2, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    # Confusion Matrix Conditions
    if (agree1 == conMx == True) and (agree2 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_naive.plot(cmap="Blues")
        plt.title("Confusion Matrix of Naive Bayes Model")
        cmCol1.pyplot()

    elif (agree2 == conMx == True) and (agree1 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_dt.plot(cmap="Blues")
        plt.title("Confusion Matrix of Decision Tree Model")
        cmCol2.pyplot()

    elif (agree3 == conMx == True) and (agree1 == agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_knn.plot(cmap="Blues")
        plt.title("Confusion Matrix of  KNN Model")
        cmCol3.pyplot()

    elif (agree1 == agree2 == conMx == True) and (agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_naive.plot(cmap="Blues")
        plt.title("Confusion Matrix of Naive Bayes Model")
        cmCol1.pyplot()

        cm_dt.plot(cmap="Blues")
        plt.title("Confusion Matrix of Decision Tree Model")
        cmCol2.pyplot()

    elif (agree2 == agree3 == conMx == True) and (agree1 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_dt.plot(cmap="Blues")
        plt.title("Confusion Matrix of Decision Tree Model")
        cmCol2.pyplot()

        cm_knn.plot(cmap="Blues")
        plt.title("Confusion Matrix of  KNN Model")
        cmCol3.pyplot()

    elif (agree1 == agree3 == conMx == True) and (agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        cm_naive.plot(cmap="Blues")
        plt.title("Confusion Matrix of Naive Bayes Model")
        cmCol1.pyplot()

        cm_knn.plot(cmap="Blues")
        plt.title("Confusion Matrix of  KNN Model")
        cmCol3.pyplot()

    elif agree1 and agree2 and agree3 and conMx:
        fig = plt.figure(figsize=(8, 8))
        cm_naive.plot(cmap="Blues")
        plt.title("Confusion Matrix of Naive Bayes Model")
        cmCol1.pyplot()

        cm_dt.plot(cmap="Blues")
        plt.title("Confusion Matrix of Decision Tree Model")
        cmCol2.pyplot()

        cm_knn.plot(cmap="Blues")
        plt.title("Confusion Matrix of KNN Model")
        cmCol3.pyplot()

# -------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ Text Dataset ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
elif Mode == 'Document Classification Dataset':
    agree1 = col1.checkbox('K-mean')
    col1.write("")
    em = chBxCol1.checkbox('Evaluation Metrics')
    Clusters = chBxCol2.checkbox('Clusters')

    cmCol1, cmCol2 = st.columns([2, 2])

    if agree1 == Clusters == True:
        image = Image.open('./Images/App images/clusters.png')
        col2.image(image)

    if agree1 == em == True:
        image = Image.open('./Images/App images/ev.png')
        cmCol1.image(image)

        image = Image.open('./Images/App images/cm.png')
        cmCol2.image(image)

# -------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ House Dataset ----------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
elif Mode == 'Platform Price Prediction Dataset':
    acc_house_lr, acc_house_dt, acc_house_knn, em_lr, em_dt, em_knn = Datasets.house()
    acc = [acc_house_lr, acc_house_dt, acc_house_knn]
    labels = ['Linear Regression', 'Decision Tree', 'KNN']

    agree1 = col1.checkbox('Linear Regression')
    agree2 = col1.checkbox('Decision Tree')
    agree3 = col1.checkbox('KNN')
    col1.write("")
    evMx = chBxCol1.checkbox('Evaluation Metrics')
    Accuracy = chBxCol2.checkbox('Accuracy')

    emCol1, emCol2, emCol3 = st.columns([2, 2, 2])

    # Accuracy Conditions
    if (agree1 == Accuracy == True) and (agree2 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[0], acc[0])
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree2 == Accuracy == True) and (agree1 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[1], acc[1])
        plt.text(0, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree3 == Accuracy == True) and (agree1 == agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[2], acc[2])
        plt.text(0, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree1 == agree2 == Accuracy == True) and (agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[0:2], acc[0:2])
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree2 == agree3 == Accuracy == True) and (agree1 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels[1:], acc[1:])
        plt.text(0, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.text(1, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif (agree1 == agree3 == Accuracy == True) and (agree2 == False):
        labels2 = ['Linear Regression', 'KNN']
        acc2 = [acc_house_lr, acc_house_knn]
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels2, acc2)
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    elif agree1 and agree2 and agree3 and Accuracy:
        fig = plt.figure(figsize=(8, 8))
        plt.bar(labels, acc)
        plt.text(0, acc[0], str(round(acc[0], 5)), ha='center', va='bottom')
        plt.text(1, acc[1], str(round(acc[1], 5)), ha='center', va='bottom')
        plt.text(2, acc[2], str(round(acc[2], 5)), ha='center', va='bottom')
        plt.ylabel('Accuracy')
        col2.pyplot(fig)

    # Evaluation Metrics Conditions
    metrics = ['MSE', 'RMSE', 'MAE', 'R-squared']

    if (agree1 == evMx == True) and (agree2 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_lr)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Linear Regression Model')
        plt.text(0, em_lr[0], str(round(em_lr[0], 5)), ha='center', va='bottom')
        plt.text(1, em_lr[1], str(round(em_lr[1], 5)), ha='center', va='bottom')
        plt.text(2, em_lr[2], str(round(em_lr[2], 5)), ha='center', va='bottom')
        plt.text(3, em_lr[3], str(round(em_lr[3], 5)), ha='center', va='bottom')
        emCol1.pyplot()

    elif (agree2 == evMx == True) and (agree1 == agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_dt)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Decision Tree Model')
        plt.text(0, em_dt[0], str(round(em_dt[0], 5)), ha='center', va='bottom')
        plt.text(1, em_dt[1], str(round(em_dt[1], 5)), ha='center', va='bottom')
        plt.text(2, em_dt[2], str(round(em_dt[2], 5)), ha='center', va='bottom')
        plt.text(3, em_dt[3], str(round(em_dt[3], 5)), ha='center', va='bottom')
        emCol2.pyplot()

    elif (agree3 == evMx == True) and (agree1 == agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_knn)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for KNN Model')
        plt.text(0, em_knn[0], str(round(em_knn[0], 5)), ha='center', va='bottom')
        plt.text(1, em_knn[1], str(round(em_knn[1], 5)), ha='center', va='bottom')
        plt.text(2, em_knn[2], str(round(em_knn[2], 5)), ha='center', va='bottom')
        plt.text(3, em_knn[3], str(round(em_knn[3], 5)), ha='center', va='bottom')
        emCol3.pyplot()

    elif (agree1 == agree2 == evMx == True) and (agree3 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_lr)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Linear Regression Model')
        plt.text(0, em_lr[0], str(round(em_lr[0], 5)), ha='center', va='bottom')
        plt.text(1, em_lr[1], str(round(em_lr[1], 5)), ha='center', va='bottom')
        plt.text(2, em_lr[2], str(round(em_lr[2], 5)), ha='center', va='bottom')
        plt.text(3, em_lr[3], str(round(em_lr[3], 5)), ha='center', va='bottom')
        emCol1.pyplot()

        plt.bar(metrics, em_dt)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Decision Tree Model')
        plt.text(0, em_dt[0], str(round(em_dt[0], 5)), ha='center', va='bottom')
        plt.text(1, em_dt[1], str(round(em_dt[1], 5)), ha='center', va='bottom')
        plt.text(2, em_dt[2], str(round(em_dt[2], 5)), ha='center', va='bottom')
        plt.text(3, em_dt[3], str(round(em_dt[3], 5)), ha='center', va='bottom')
        emCol2.pyplot()

    elif (agree2 == agree3 == evMx == True) and (agree1 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_dt)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Decision Tree Model')
        plt.text(0, em_dt[0], str(round(em_dt[0], 5)), ha='center', va='bottom')
        plt.text(1, em_dt[1], str(round(em_dt[1], 5)), ha='center', va='bottom')
        plt.text(2, em_dt[2], str(round(em_dt[2], 5)), ha='center', va='bottom')
        plt.text(3, em_dt[3], str(round(em_dt[3], 5)), ha='center', va='bottom')
        emCol2.pyplot()

        plt.bar(metrics, em_knn)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for KNN Model')
        plt.text(0, em_knn[0], str(round(em_knn[0], 5)), ha='center', va='bottom')
        plt.text(1, em_knn[1], str(round(em_knn[1], 5)), ha='center', va='bottom')
        plt.text(2, em_knn[2], str(round(em_knn[2], 5)), ha='center', va='bottom')
        plt.text(3, em_knn[3], str(round(em_knn[3], 5)), ha='center', va='bottom')
        emCol3.pyplot()

    elif (agree1 == agree3 == evMx == True) and (agree2 == False):
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_lr)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Linear Regression Model')
        plt.text(0, em_lr[0], str(round(em_lr[0], 5)), ha='center', va='bottom')
        plt.text(1, em_lr[1], str(round(em_lr[1], 5)), ha='center', va='bottom')
        plt.text(2, em_lr[2], str(round(em_lr[2], 5)), ha='center', va='bottom')
        plt.text(3, em_lr[3], str(round(em_lr[3], 5)), ha='center', va='bottom')
        emCol1.pyplot()

        plt.bar(metrics, em_knn)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for KNN Model')
        plt.text(0, em_knn[0], str(round(em_knn[0], 5)), ha='center', va='bottom')
        plt.text(1, em_knn[1], str(round(em_knn[1], 5)), ha='center', va='bottom')
        plt.text(2, em_knn[2], str(round(em_knn[2], 5)), ha='center', va='bottom')
        plt.text(3, em_knn[3], str(round(em_knn[3], 5)), ha='center', va='bottom')
        emCol3.pyplot()

    elif agree1 and agree2 and agree3 and evMx:
        fig = plt.figure(figsize=(8, 8))
        plt.bar(metrics, em_lr)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Linear Regression Model')
        plt.text(0, em_lr[0], str(round(em_lr[0], 5)), ha='center', va='bottom')
        plt.text(1, em_lr[1], str(round(em_lr[1], 5)), ha='center', va='bottom')
        plt.text(2, em_lr[2], str(round(em_lr[2], 5)), ha='center', va='bottom')
        plt.text(3, em_lr[3], str(round(em_lr[3], 5)), ha='center', va='bottom')
        emCol1.pyplot()

        plt.bar(metrics, em_dt)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for Decision Tree Model')
        plt.text(0, em_dt[0], str(round(em_dt[0], 5)), ha='center', va='bottom')
        plt.text(1, em_dt[1], str(round(em_dt[1], 5)), ha='center', va='bottom')
        plt.text(2, em_dt[2], str(round(em_dt[2], 5)), ha='center', va='bottom')
        plt.text(3, em_dt[3], str(round(em_dt[3], 5)), ha='center', va='bottom')
        emCol2.pyplot()

        plt.bar(metrics, em_knn)
        plt.ylabel('Metric Value')
        plt.title('Evaluation Metrics for KNN Model')
        plt.text(0, em_knn[0], str(round(em_knn[0], 5)), ha='center', va='bottom')
        plt.text(1, em_knn[1], str(round(em_knn[1], 5)), ha='center', va='bottom')
        plt.text(2, em_knn[2], str(round(em_knn[2], 5)), ha='center', va='bottom')
        plt.text(3, em_knn[3], str(round(em_knn[3], 5)), ha='center', va='bottom')
        emCol3.pyplot()