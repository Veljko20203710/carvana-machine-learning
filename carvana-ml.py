#!/usr/bin/env python
# coding: utf-8

# In[227]:


#importovanje potrebnih biblioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
from scipy import stats
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None


# In[228]:


#citanje csv fajla sa separatorom ;
carvana = pd.read_csv('carvana.csv',sep=',')


# In[229]:


#kada izvrsimo funkciju info() dobijamo tipove vrednosti za svaku kolonu. Vidimo da skup podataka ima kolone tipa object,int64 i float64
carvana.info()
carvana.describe()


# In[230]:


#vrsimo deskripciju kategorickih promenjihiv, vidimo da model, submodel, trim i purchdate imaju veliki broj unique vrednosti, sto nije dobro. Treba da razmotrimo da ih izbacimo iz modela
carvana.describe(include=[np.object])


# In[231]:


#funkcijom isna().sum() vidimo ukupan broj nepoznatih vrednosti po kolonama.
#Auction, primeunit, aucguart i vnst imaju ogroman broj NA vrednosti, pa ce zbog toga biti izbaceni iz modela,
#jer se NA vrednosti ne mogu zameniti nekim drugim, dominantim vrednostima.
carvana.isna().sum()


# In[232]:


#kolone wheelType i wheelTypeID su iste za sve instance, takve su redudante i kolona wheelType se moze izbrisati
del carvana['WheelType']


# In[233]:


#kolona purchDate sadrzi mnogo unique vrednosti pa treba da je izbacimo
del carvana['PurchDate']


# In[234]:


#Veliki broj unique vrednosti za kategoricke varijable model, trim, submodel nije dobar pa mozemo da pokusamo da ih obrisemo
del carvana['Trim']
del carvana['Model']
del carvana['SubModel']


# In[235]:


#Vidimo da je veliki broj NA vrednosti u Auction, PRIMEUNIT, AUCGUART i VNST kolonama
#Posto je veliki broj NA vrednosti ove kolone mozemo da izbacimo iz daljeg razmatranja
del carvana['VNST']
del carvana['Auction']
del carvana['PRIMEUNIT']
del carvana['AUCGUART']


# In[236]:


#Sada vidimo novi broj na vrednosti
carvana.isna().sum()


# In[237]:


#ovde vidimo da ovaj red ima NA vrednosti za NA i Size pa cemo ga izbaciti
carvana[carvana['Size'].isna()]


# In[238]:


#Ovde vidimo da ova 21 reda imaju NA vrednosti za MMRCurrentAuctionAveragePrice,MMRCurrentAuctionCleanPrice,MMRCurrentRetailAveragePrice,MMRCurrentRetailCleanPrice 
carvana[carvana['MMRCurrentAuctionAveragePrice'].isna()]


# In[239]:


#Ovde vidimo da je tip 1 i tip 2 guma imaju slican broj pojavljivanja, pa ne mozemo da odredimo dominantu klasu.
#Da smo mogli da odredimo dominantu, NA vrednosti bi bile zamenjene sa njom a ovako ce NA vrednosti biti obrisane
sn.catplot(x='WheelTypeID', kind="count", palette="ch:.25",data=carvana);


# In[240]:


carvana.info()


# In[241]:


carvana.isna().sum()


# In[242]:


#uklanjenjem NA vrednosti broj redova se smanjio za 305 (283 koji su imali NA u WheelTypeID-u, 21 sa MMRCurrentAuctionAveragePrice,
#MMRCurrentAuctionCleanPrice, MMRCurrentRetailAveragePrice, MMRCurrentRetailCleanPrice i jedna za ostale NA vrednosti)
carvana = carvana.dropna()


# In[243]:


#Vidimo da vise nemamo NA vrednosti
carvana.isna().sum()


# In[244]:


#numericka korealaciona matrica
#Visoko korelisani atributi su oni atributi cena (MMRAcquisitionAuctionAveragePrice,MMRAcquisitionAuctionCleanPrice,MMRAcquisitionRetailAveragePrice,MMRAcquisitonRetailCleanPrice,MMRCurrentRetailCleanPrice)
carvana.corr()
#korelaciona matrica sa brojevima i bojama u zavisnosti od koeficijenta korelacije
corr = carvana.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[245]:


#iz matrice korelacije vidimo da su sve promenjive sa cenom (Price) visoko korelisane
#moramo razmotriti mogucnost da eliminisemo sve osim jedne jer dolazi do redudanse podataka
del carvana['MMRCurrentRetailAveragePrice']
del carvana['MMRCurrentAuctionCleanPrice']
del carvana['MMRCurrentAuctionAveragePrice']
del carvana['MMRAcquisitonRetailCleanPrice']
del carvana['MMRAcquisitionRetailAveragePrice']
del carvana['MMRAcquisitionAuctionCleanPrice']
del carvana['MMRAcquisitionAuctionAveragePrice']


# In[246]:


carvana.reset_index(drop=True)


# In[247]:


#pravljenje ulaza i izlaza modela
X = carvana.iloc[:,:-1]
y = carvana.iloc[:,-1]


# In[248]:


#kreiranje dummies promenjivih od kategorickih varijabli
categorical = ['Make','Color','Transmission','Nationality','Size','TopThreeAmericanName']
pom = X.select_dtypes(include=['int64','float64'])
for i in categorical:
    X[i] = X[i].astype('category')
    dummy = pd.get_dummies(X[i])
    pom = pd.concat([pom,dummy],axis=1)
X = pom
X.info()


# In[249]:


#prikaz svih atributa koji imaju varijansu preko 0.05. 
from sklearn.feature_selection import VarianceThreshold

selection = VarianceThreshold(0.05)
selection.fit(X)

print(selection.get_support())
print('--------------------')
print(selection.get_support().sum())


# In[250]:


X


# In[251]:


#Eliminisanjem kolona koja imaju varijansu manju od 0.05
X = X.loc[:, selection.get_support()]


# In[252]:


X


# In[253]:


#Metoda izbora 25 atribiuta koja najvise nose informacije
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

selection = SelectKBest(k=25, score_func=mutual_info_classif)  

selection.fit(X, y)
selection.get_support()
X = X.loc[:, selection.get_support()]


# In[254]:


#funkcija za prikaz ocene tacnosti modela
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

def print_performance(y_parametar, y_hat):
    print(f'Accuracy: {accuracy_score(y_parametar, y_hat)}')
    print(f'Precision: {precision_score(y_parametar, y_hat)}')
    print(f'Recall: {recall_score(y_parametar, y_hat)}')
    print(f'F1: {f1_score(y_parametar, y_hat)}')


# In[255]:


#Funkcija koja racuna ocene tacnosti modela koju sam napisao zbog promene pozitivne klase
def print_performance_from_confusion_matrix(confusion_matrix):
    print(f'Accuracy: {(confusion_matrix[0][0]+confusion_matrix[1][1])/((confusion_matrix[0][0]+confusion_matrix[1][1])+confusion_matrix[0][1]+confusion_matrix[1][0])}')
    precision = (confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[0][1])
    print(f'Precision: {precision}')
    recall = (confusion_matrix[0][0])/(confusion_matrix[0][0]+confusion_matrix[1][0])
    print(f'Recall: {recall}')
    print(f'F1: {2*precision*recall/(precision+recall)}')


# In[256]:


#funkcija za ocenu modela samo na trening setu
#accurarcy se dobija kada se broj tacnih predvidjenih podeli sa brojem ukupnih predvidjanja (kolicnik zbira na glavnoj 
#dijagonali matrice konfuzije i zbira svih elemenata matrice konfuzije). On pokazuje koliko je nas model tacan.
#precison je kolicnik zbira True pozivno predvidjenih sa ukupno True predvidjenim. Odgovora na pitanje koliki je procenatac tacno predvidjenih ustvari tacan
#recall odgovora na pitanje koliko je stvarnih true positive predvidjeno modelom. Kod nekih sistema, kao sto je na primer otkrivanje bolesti, je potrebno
#da ovaj model ima sto vecu vrednost.
#precision i recall su suprotne vrednosti, kada jedna opada druga raste i obrnuto
#f1 oznacava meru izmedju precision i recall.
from sklearn.metrics import confusion_matrix
def ocena_trening(algoritam,X_parametar,y_parametar):
    algoritam.fit(X_parametar,y_parametar)
    y_hat = algoritam.predict(X_parametar)
    print_performance_from_confusion_matrix(confusion_matrix(y_true=y_parametar,y_pred=algoritam.predict(X)))
    print(confusion_matrix(y_true=y_parametar,y_pred=algoritam.predict(X)))


# In[257]:


#funkcija koja deli podatke na trening i test set u proporciji 70:30, zatim kreira model na osnovu treninga, predvidja rezultate test podataka i upordjuje
#ovakve izlaze sa stvarnim izlazima u matrici konfuzije
from sklearn.model_selection import train_test_split
def ocena_test(algoritam,X_parametar,y_parametar):
    X_train, X_test,y_train, y_test = train_test_split(X_parametar,y_parametar,test_size=0.3,random_state=2020)
    algoritam.fit(X_train,y_train)
    y_hat = algoritam.predict(X_test)
    print_performance_from_confusion_matrix(confusion_matrix(y_true=y_parametar,y_pred=algoritam.predict(X)))
    confusion_matrix(y_true=y_test,y_pred=algoritam.predict(X_test))
    print(confusion_matrix(y_true=y_test,y_pred=algoritam.predict(X_test)))


# In[258]:


#funkcija koja pomera granicu odlucivanja (smanjenjem granice se povecava recall a smanjuje precision)
from sklearn.metrics import confusion_matrix
def granica_odlucivanja(algoritam,granica,X_parametar,y_paramatar):
    algoritam.fit(X,y)
    y_hat = algoritam.predict_proba(X)[:, 1] >= granica
    print('===== granica '+str(granica)+"=======")
    print_performance_from_confusion_matrix(confusion_matrix(y_true=y,y_pred=y_hat))
    print(confusion_matrix(y,y_hat))


# In[259]:


#funkcija koja radi cross-validaciju podelom na 10 delova, za svaki tako dobijen model prikazuje tacnost i na kraju prikazuje tacnost citave cross-validatcije
from sklearn.model_selection import KFold
def cross_validation(algoritam,X_parametar,y_parametar):
    folds = KFold(n_splits=10)
    results = []
    for train_index, test_index in folds.split(X_parametar):
        X_train, y_train  = X_parametar.loc[train_index, :], y_parametar[train_index]
        X_test, y_test = X_parametar.loc[test_index], y_parametar[test_index]
    
        X_train = X_train.dropna()
        y_train = y_train.dropna()
        X_test = X_test.dropna()
        y_test = y_test.dropna()
        
        algoritam.fit(X_train, y_train)
    
        y_hat = algoritam.predict(X_test)
        results.append(accuracy_score(y_test, y_hat))
        print(f'Accuracy: {accuracy_score(y_test, y_hat)}')
    print(f'Tačnost iznosi {round(np.mean(results) * 100, 2)}% +/- {round(np.std(results)*100, 2)}%')


# In[260]:


#funkcija koja iscrtava ROC AUC krivu i racuna povrsinu ispod nje
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
def roc(algoritam,X_parametar,y_parametar):
    algoritam.fit(X,y)
    print('AUC vrednost je '+str(roc_auc_score(y_parametar,algoritam.predict_proba(X_parametar)[:,1])))
    result = cross_val_score(algoritam,X_parametar,y_parametar,cv=10,scoring='roc_auc')
    print('Cross validation ROC='+str(result.mean())+"+-"+str(result.std()))
    fpr, tpr, thresholds = roc_curve(y, algoritam.predict_proba(X_parametar)[:, 1])
    plt.plot(fpr, tpr)


# In[261]:


#Vidimo da je tacnost dobra 0.89 dok su i ostale statistike dobre medjutim model predvidja samo pozitivnu klasu sto nije dobro.
from sklearn.linear_model import LogisticRegression
ocena_trening(LogisticRegression(),X,y)


# In[262]:


#Ovde smanjujemo granicu odlucivanja da bismo dobili i predvidjana negativne klase
#Ovde vidimo da se accuracy neznatno dok se precions takodje smanjio ali se odziv nezantno smanjio
granica_odlucivanja(LogisticRegression(),0.2,X,y)


# In[263]:


#Mogli bismo da probamo i dodatno da smanjimo granicu odlucivanja
#sada vidimo veliki broj predividjana negativne klase, tacnost i preciznost i F1 statistika su se znacajno smanjile dok se odziv povecao
granica_odlucivanja(LogisticRegression(),0.1,X,y)


# In[264]:


#Mogli bismo da probamo da nacrtamo AUC krivu 


# In[265]:


#Ovde vidimo da povrsina ispod krive je 0.63 sto i nije bas najbolja vrednost
roc(LogisticRegression(),X,y)


# In[266]:


#Vrsimo regularizaciju logisticke regresije, regularizacijom se sprecava overfitting 
#C je paramaetar regularazijacije
from sklearn.model_selection import GridSearchCV
logistic_params = {'C':np.linspace(start=0.001, stop=2, num=30)}
grid = GridSearchCV(LogisticRegression(), logistic_params, cv=10, scoring='roc_auc')
grid.fit(X,y)

print('Best param: ', grid.best_params_)


# In[267]:


#Sve cetiri statistike imaju visoku vrednost
from sklearn.neighbors import KNeighborsClassifier
ocena_trening(KNeighborsClassifier(),X,y)


# In[268]:


#ovde su performanse modela neznatno opale u odnosu na model sa samo trening setom
ocena_test(KNeighborsClassifier(),X,y)


# In[269]:


#Smanjujemo granicu odlucivanja na 0.3 Smanjenem granice odlucivanja odziv se povecao dok su se ostale statistike smanjile
granica_odlucivanja(KNeighborsClassifier(),0.3,X,y)


# In[270]:


#tacnosti je slicna kao i na prethodnom modelu bez cross-validacije
cross_validation(KNeighborsClassifier(),X,y)


# In[271]:


#Ovde vidimo da je vrednost ROC odlicna
roc(KNeighborsClassifier(),X,y)


# In[272]:


#Vrsimo kombinaciju provere razlicitih hiperametara, broj grupa i tipova distanci. 
params_k = [1, 3, 5, 7, 9, 11, 13, 15]
params_dist = ['euclidean', 'manhattan', 'chebyshev']

for k in params_k:
    for dist in params_dist:
        model = KNeighborsClassifier(n_neighbors=k, metric=dist)
        print(f'K={k}, Distance={dist} - AUC: {cross_val_score(model, X, y, cv=10, scoring="accuracy").mean()}')


# In[273]:


#Perfomanse kod drveta odlucivanja su odlicne sto je posledice overffiting-a
from sklearn.tree import DecisionTreeClassifier
ocena_trening(DecisionTreeClassifier(),X,y)


# In[274]:


#ovde su performanse modela znatno opale u odnosu na model sa samo trening setom. U trening setu model je u potpunosti bio tacan medjutim to je posledica overffiting-a
ocena_test(DecisionTreeClassifier(),X,y)


# In[275]:


#rezultati su znacajno manji nego na model sa samo trening skupom podataka i na model podataka sa trening i test skupom.
cross_validation(DecisionTreeClassifier(),X,y)


# In[276]:


#Ovde vidimo da je vrednost ROC odlicna
roc(DecisionTreeClassifier(),X,y)


# In[277]:


#Vrsimo kombinaciju provere razlicitih hiperametara, maksimalan broj listova, broj primera potrebnih za deljenje, kriterijum granjanja 
#Vidimo da vrednsosti sa malim brojem max_leaf_nodes od 2 i 3 bez obzira na ostala dva hiperparametra
max_leaf_nodes = [2, 3, 5, 7, 9, 11, 13, 15]
min_samples_leaf = [1, 3, 5, 7, 9, 11, 13, 15]
criterion = ['gini','entropy']

for k in max_leaf_nodes:
    for n in min_samples_leaf:
        for c in criterion:
            model = DecisionTreeClassifier(criterion=c, max_leaf_nodes=k, min_samples_leaf=n)
            print(f'max_leaf_nodes={k}, min_samples_leaf={n}, criterion={c} - AUC: {cross_val_score(model, X, y, cv=10, scoring="accuracy").mean()}')


# In[278]:


from sklearn.naive_bayes import GaussianNB
ocena_trening(GaussianNB(),X,y)


# In[279]:


#ovde su performanse modela slicne kao i model sa samo trening setom
ocena_test(GaussianNB(),X,y)


# In[280]:


#ovde je tacnost znacajno slicna kao i model bez cross-validacije
cross_validation(GaussianNB(),X,y)


# In[281]:


#Ovde vidimo da povrsina ispod krive je 0.64 sto i nije bas najbolja vrednost
roc(GaussianNB(),X,y)


# In[282]:


#Mozemo da pokusamo sa algoritmima ansambilima. Prvi takav algoritam je Random Forest


# In[283]:


#Ovaj algoritam pokazuje sjajne rezultate
from sklearn.ensemble import RandomForestClassifier
ocena_trening(RandomForestClassifier(),X,y)


# In[284]:


#Rezultati su i dalje odlicni ali malo slabiji
ocena_test(RandomForestClassifier(),X,y)


# In[285]:


#Tacnost je opala
cross_validation(RandomForestClassifier(),X,y)


# In[286]:


#AUC vrednost je skoro 1
roc(RandomForestClassifier(),X,y)


# In[287]:


#Ovde radimo GridSeacrhCV kojim racunamo performanse za kombinacije hiperparametara.
#Hiperpamateri koji ulazi su maksimalna dubina stabla, najmanji broj listova, maksimalan broj atributa u drvetu 
forest = RandomForestClassifier(n_estimators=10, n_jobs=None, random_state=42, class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9],'max_depth': [5,10,15]}
grid = GridSearchCV(forest, parameters, cv=5, scoring='accuracy')
grid.fit(X,y)
print('Best param: ', grid.best_params_)


# In[288]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


# In[289]:


#Vidimo da se tacnost ne poboljsava uvodjenjem glasanja
model_nb = RandomForestClassifier()
model_lr = LogisticRegression()

model = VotingClassifier(voting='soft', estimators=[('knn', model_nb), ('lr', model_lr)])

for model, label in zip([model_nb, model_lr, model], ['knn', 'LR', 'Voting']):
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f'{label}: {scores.mean()}')


# In[290]:


#Pokusavamo da soft glasanjem tri algoritma logistisckom regresiojom, naivnim bajesom i knn dobijemo bolji rezul
#Tacnost se neznatno poboljsala
model_lr = RandomForestClassifier()
model_nb = GaussianNB()
model_knn = KNeighborsClassifier()

model = VotingClassifier(voting='soft', 
                         estimators=[
                                    ('RF', model_lr),
                                    ('NB', model_nb),
                                    ('knn', model_knn)])

for model, label in zip([model_lr, model_nb, model_knn, model], ['LR', 'NB', 'KNN', 'Voting']):
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f'{label}: {scores.mean()}')


# In[291]:


#Ovde kod stacking-a vidimo vidimo da se vrednost asembli algortima neznatno popravila
from mlxtend.classifier import StackingClassifier

model_knn = KNeighborsClassifier()
model_nb = GaussianNB()

model_stacking = StackingClassifier(classifiers=[model_knn, model_nb], meta_classifier = model_knn)

scores = cross_val_score(model_knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

scores = cross_val_score(model_nb, X, y, cv=10, scoring='accuracy')
print(scores.mean())

scores = cross_val_score(model_stacking, X, y, cv=10, scoring='accuracy')
scores.mean()


# In[292]:


# Ukljucivanje svih potrebnih biblioteka

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# In[293]:


from keras.metrics import AUC
from sklearn.model_selection import cross_val_score
#funkcija koja pravi i validira neuronsku mrezu. Ulazni podaci su aktivaciona funkcija, broj slojeva, broj neurona po slojevima i izlazni parametri 
def neural_network(activation_function,number_of_layers,number_of_neuron_per_leyers,y):
    folds = KFold(n_splits=10)  #Pravljenje 10 foldova za krosvalidaciju
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train_index, test_index in folds.split(X):
        
        #podela na trening i test set na osnovu krosvalidacije
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        #kreiranje sekvencijalnog modela neuronske mreze
        model = Sequential()
        
        #dodavanje ulaznog sloja
        model.add(Dense(number_of_neuron_per_leyers, input_shape=(25,), activation=activation_function))
        
        #dodovanje skrivenih slojeva neuronske mreze
        for i in range (1,number_of_layers):
            model.add(Dense(number_of_neuron_per_leyers, activation=activation_function))
   
        #dodavanje izlaznog sloja
        model.add(Dense(1,))
        
        #Pravljenje modela sa Adam algoritmom za optimizaciju, learning ratom od 0.003 i metrikom za evalucaiju srednja kvadratna greska
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])   

        #Kreiranje kontrole funkcije koja zauzstavlja model u trenutku u kom model prestaje da se poboljstava
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        # Vrsi treniranje modela
        model.fit(X_train, y_train, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, 
                    callbacks = [earlystopper])
          
        y_test_pred = model.predict_classes(X_test)
        print_performance_from_confusion_matrix(confusion_matrix(y_true=y_test,y_pred=y_test_pred))


# In[294]:


neural_network("sigmoid",1,1,y)


# In[295]:


neural_network("sigmoid",2,2,y)


# In[296]:


neural_network("sigmoid",10,10,y)


# In[297]:


neural_network("relu",1,1,y)


# In[298]:


neural_network("relu",10,10,y)


# In[299]:


neural_network("tanh",1,1,y)


# In[300]:


neural_network("tanh",10,10,y)

