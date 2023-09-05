# ITI-AI-Final-Project

## Table of contents:

- [Introduction](#introduction)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [How to Run The Project](#run-the-project)
- [Team](#team)

### Introduction
This website provides the user with the ability to see the accuracies of AI models and evaluation matrices of the 'Email Spam Detection Dataset', 'Document Classification Dataset', and 'Platform Price Prediction Dataset' datasets.
<p align="center">
<img src="Images/Readme images/website.png" style="border-radius: 50%;" /> 
</p>


### Project Features
1. Choose Dataset
<p align="center">
<img src="Images/Readme images/choose.png" style="border-radius: 50%;" /> 
</p>
2. Choose Models
<p align="center">
<img src="Images/Readme images/models.png" style="border-radius: 50%;" /> 
</p>
3. Choose Accuracy
<p align="center">
<img src="Images/Readme images/acc.png" style="border-radius: 50%;" /> 
</p>
4. Choose Evaluation Matrices
<p align="center">
<img src="Images/Readme images/ev.png" style="border-radius: 50%;" /> 
</p>
5. Show Accuracies of Models
<p align="center">
<img src="Images/Readme images/show_acc.png" style="border-radius: 50%;" /> 
</p>
6. Show Evaluation Matrices of Models
<p align="center">
<img src="Images/Readme images/show_ev.png" style="border-radius: 50%;" /> 
</p>

### Project Structure
The Web Application is built using:

<p align="left">
<!-- *Streamlit -->
<a href="https://streamlit.io" target="_blank" rel="noreferrer"> <img src="Images/Readme images/streamlit.svg" alt="tensorflow" width="120" height="40"/> </a>
<!-- python -->
<a href="https://www.python.org" target="_blank"><img align="center" alt="Python" height ="48px" width="48" src="https://raw.githubusercontent.com/rahul-jha98/github_readme_icons/main/language_and_tools/square/python/python.svg"></a>
<!-- *SKlearn -->
<a href="https://scikit-learn.org/stable/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="tensorflow" width="60" height="40"/> </a>
<!-- *Pandas -->
<a href="https://pandas.pydata.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="tensorflow" width="80" height="40"/> </a>
<!-- *Numpy -->
<a href="https://numpy.org" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg" alt="tensorflow" width="80" height="40"/> </a>
<!-- *Matplotlib -->
<a href="https://matplotlib.org" target="_blank" rel="noreferrer"> <img src="Images/Readme images/matplot.png" alt="tensorflow" width="120" height="40"/> </a>
<!-- *seaborn -->
<a href="https://seaborn.pydata.org" target="_blank" rel="noreferrer"> <img src="Images/Readme images/Seaborn.png" alt="tensorflow" width="120" height="40"/> </a>
<!-- *plotly -->
<a href="https://plotly.com" target="_blank" rel="noreferrer"> <img src="Images/Readme images/plotly.png" alt="tensorflow" width="120" height="40"/> 
</a>

</p>

- Frontend:
  - Streamlit
  - CSS
- Backend:
  - Python
  
The frontend it's main function is to set the structure of the page while the backend consists of the pages of classes of models and functions of preprocessing.

```
main
├─ App Folder
|  ├─  main.py
│  ├─  datasets.py
│  ├─  models.py
|  ├─  preprocessing.py
|  ├─  HousingData.csv
|  ├─  spam_ham_dataset.csv
│  └─  sarcasm.json
├─ Images Folder
├─ App Folder
|  ├─  App images Folder
│  └─  Readme images Folder
├─ Notebooks Folder
|  ├─  House.ipynb
│  ├─  spamEmailsDetection.ipynb
│  ├─  TextClustring.ipynb
|  ├─  HousingData.csv
|  ├─  spam_ham_dataset.csv
│  └─  sarcasm.json
└─ README.md
```

### Run the Project

1. Install Python3 on your computer

```
Download it from www.python.org/downloads/
```

2. Install the following packages
```
pip install streamlit
```
```
pip install pybase64
```
```
pip install numpy
```
```
pip install pandas
```
```
pip install scikit-learn
```
```
pip install sklearn
```
```
pip install matplotlib
```
```
pip install Pillow
```
```
pip install image
```
```
pip install standard-scaler
```
```
pip install StandardScaler
```
```
pip install LinearRegression
```
```
pip install LogisticRegression-PavanAdapa
```
```
pip install SVC
```
```
pip install PyLMNN
```
```
pip install naive-bayes
```
```
pip install DecisionTree
```
```
pip install kmeans
```
```
pip install fastcountvectorizer
```
```
pip install sklearn-features
```
```
pip install metrics
```
```
pip install tfidf
```
```
pip install pca
```


3. Open Project Terminal & Run

```
pip install -r requirments.txt
```

4. Start the Server by Running and writing in the terminal:

```
streamlit run App/main.py
```

5. Visit by clicking on the terminal result link:
```
Local URL: http://localhost:8501
Network URL: http://192.168.1.2:8501
```

### Team

| Team Members' Names                                      | 
| -------------------------------------------------------- | 
| [Dina Hussam](https://github.com/Dinahussam)             |   
| [Asala Ehab](https://github.com/Asalaehab)               |    
| [Mostafa Ali Abdulhalim](https://github.com/Mostafaali3) |     

### Submitted to:

- Dr. Salwa Osama
