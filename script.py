print('Importando librerias...')
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier

print('Consiguiendo los datos de entrenamiento y el conjunto de prueba...')
train = pd.read_csv("train.csv", dtype={"Age": np.float64})
test  = pd.read_csv("test.csv", dtype={"Age": np.float64})

print('Datos de entrenamiento:', len(train))

print('Datos de prueba:', len(test))

print('Cleaning the dataset...')




def harmonize_data(titanic):
    # Llena campos vacíos
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # Asigna valores numéricos a los datos para facilitar cálculos
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 0
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    return titanic

print('Creando submission file...')    
def create_submission(dtc, train, test, predictors, filename):
    dtc.fit(train[predictors], train["Survived"])
    predictions = dtc.predict(test[predictors])
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)

print('Limpiando datos...')    
train_data = harmonize_data(train)
test_data  = harmonize_data(test)

print('Performing feature enginnering...') 
train_data["PSA"] = train_data["Pclass"]*train_data["Sex"]*train_data["Age"]
train_data["SP"] = train_data["SibSp"]+train_data["Parch"]
test_data["PSA"] = test_data["Pclass"]*test_data["Sex"]*test_data["Age"]
test_data["SP"] = test_data["SibSp"]+test_data["Parch"]

print('Definiendo predictores...')
#Clase de pasajero, sexo, edad, PSA*, pago del 
#pasajero, lugar de embarcación y SP (no. de familiares a bordo)
predictors = ["Pclass", "Sex", "Age", "PSA", "Fare", "Embarked", "SP"]

print('Encontrando la mejor max_depth para DecisionTreeClassifier(clasificador de arbol de decisiones)...')
max_score = 0
best_n = 0
for n in range(1,100):
    dtc_scr = 0.
    dtc = DecisionTreeClassifier(max_depth=n)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        dtc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
        dtc_scr += dtc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
    if dtc_scr > max_score:
        max_score = dtc_scr
        best_n = n
print(best_n, max_score)

print('Encontrando la mejor partición min_samples_splir para DecisionTreeClassifier...')
max_score = 0
best_s = 0
for s in range(1,100):
    dtc_scr = 0.
    dtc = DecisionTreeClassifier(min_samples_split=s)
    for train, test in KFold(len(train_data), n_folds=10, shuffle=True):
        dtc.fit(train_data[predictors].T[train].T, train_data["Survived"].T[train].T)
        dtc_scr += dtc.score(train_data[predictors].T[test].T, train_data["Survived"].T[test].T)/10
    if dtc_scr > max_score:
        max_score = dtc_scr
        best_s = s
print(best_n, max_score)

print('Haciendo predicciones...')
dtc = DecisionTreeClassifier(max_depth=best_n, min_samples_split=s, criterion='entropy', splitter='random')
print('Creando submission...')
create_submission(dtc, train_data, test_data, predictors, "dtcsurvivors.csv")
print('Listo.')