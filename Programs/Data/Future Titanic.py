import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# Используемые функции
def Spl(data, lst, col_name, sing):
    data[lst] = data[col_name].str.split(sing, len(lst), expand = True)
    data = data.pop(col_name)
    return data

# Загрузка данных
t_tr = pd.read_csv(r'C:\Users\shifr\OneDrive\Desktop\Programs\data\train.csv')
tit_test = pd.read_csv(r'C:\Users\shifr\OneDrive\Desktop\Programs\data\test.csv')
titsample_sub = pd.read_csv(r'C:\Users\shifr\OneDrive\Desktop\Programs\data\sample_submission.csv')

# Описание данных
print(t_tr.info())
print(t_tr.sample(5))
print(t_tr.describe(include='all'))

"""DONE"""

# Разделим строчные строки на несколько
Spl(t_tr, ['Deck', 'Room', 'Class'], 'Cabin', '/')
Spl(t_tr, ['First_name', 'Second_name'], 'Name', ' ')
Spl(t_tr, ['Group', 'G_num'], 'PassengerId', '_')


# Факторизируем наши бинаврные данные,  втом числе целевую переменную
t_tr['Transported'] = pd.factorize(t_tr['Transported'])[0]

# Заполнение пропусков 
# Создаем дополнительный столбец с потраченными деньгами
Money =['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
t_tr[Money] = t_tr[Money].fillna(0)
t_tr['Sum'] = t_tr['RoomService'] + t_tr['FoodCourt']+ t_tr['ShoppingMall']+ t_tr['Spa']            
# t_tr['Sum_no_room_service'] = t_tr['FoodCourt']+ t_tr['ShoppingMall']+ t_tr['Spa']

"""DONE"""

# Заполним Nan фичей VIP, CryoSleep

t_tr['VIP'] = pd.factorize(t_tr['VIP'])[0]

for i in range(t_tr.shape[0]):
    if t_tr.iloc[i, 4] == -1:
        if t_tr.iloc[i, 18] >= 4000.0:
            t_tr.iloc[i, 4] = 1
        else:
            t_tr.iloc[i, 4] = 0
            
t_tr['VIP'] = t_tr['VIP'].astype('object')

t_tr['CryoSleep'] = t_tr['CryoSleep'].fillna('False')
t_tr['CryoSleep'] = pd.factorize(t_tr['CryoSleep'])[0]
"""DONE"""
    
# Заполним пустые значения возраста
t_tr['Age'] = t_tr['Age'].fillna(t_tr['Age'].median())

for i in range(t_tr.shape[0]):
    if t_tr.iloc[i, 3] == 0:
       t_tr.iloc[i, 3] = t_tr['Age'].median() 
"""DONE"""

# Заполним пропуски в HomePlanet & Destination.
# С помощью sns.heatmap посмотрим наиболее вероятный пункт отбытия и прибытия.
pl = t_tr[['HomePlanet', 'Destination']]

t_tr['Destination'].map()
for i in range(t_tr.shape[0]):
    if t_tr.iloc[i, 0] == 'Europa':
        if t_tr.iloc[i, 2] == 'Nan':
            t_tr.iloc[i, 2] = '55 Cancri e'
    if t_tr.iloc[i, 0] == 'Earth':
        if t_tr.iloc[i, 2] == 'Nan':
            t_tr.iloc[i, 2] = 'PSO J318.5-22'
    if t_tr.iloc[i, 0] == 'Mars':
        if t_tr.iloc[i, 2] == 'Nan':
            t_tr.iloc[i, 2] = 'TRAPPIST-1e'        

t_tr['HomePlanet'] = t_tr['HomePlanet'].fillna('Earth')   
t_tr['Destination'] = t_tr['Destination'].fillna('TRAPPIST-1e') 

"""DONE"""

# Заполним пропуски в Deck, Room, Class
t_tr['Deck'] = t_tr['Deck'].fillna('F')
t_tr['Class'] = t_tr['Class'].fillna('D')
t_tr['Room'] = t_tr['Room'].fillna(np.random.randint(1, 1800))


y = t_tr['Transported']
r = t_tr.drop(['Second_name','First_name','Group', 'G_num', 'Room', 'Transported','Sum'], axis = 1)
col = r.columns


cat_f = [i for i in col if r[i].dtype.name == 'object']
num_f = [i for i in col if r[i].dtype.name == 'int64' or r[i].dtype.name == 'float64']

n_d = r[num_f]
c_d = r[cat_f]

print(cat_f, num_f)
dum_feat = pd.get_dummies(c_d, drop_first=True) 
X = pd.concat([n_d, dum_feat], axis = 1)
t_tr.info()

sns.set(rc={'figure.figsize':(15,12)})
sns.heatmap(X.corr(), square = True,
            annot = True, fmt ='.2g', annot_kws = dict(size = 9, weight = 'bold'),
            vmin = -1, vmax = 1,
            linewidths=1,
            linecolor='black',
            cmap = 'coolwarm').set(title = 'HEAT')

# # Анализ данных и посторение моделей
# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import make_pipeline

# X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 0)

# KNN_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())

# print(KNN_pipeline.get_params().keys())
# parameters = {'kneighborsclassifier__n_neighbors': np.array(np.linspace(4, 30, 8), dtype='int'),
#              'kneighborsclassifier__metric': ['manhattan', 'euclidean'],
#              'kneighborsclassifier__weights': ['uniform', 'distance']}
# CV_model = GridSearchCV(estimator=KNN_pipeline, 
#                         param_grid = parameters,
#                         cv=5, 
#                         scoring='roc_auc',
#                         n_jobs=-1, 
#                         verbose=10)
# knn = CV_model.fit(X_tr, y_tr)
# print(knn.best_params_)

# #y_pred_prob_KNN = knn.predict_proba(X_t)[:, 1]
# y_pred_KNN = knn.predict(X_t)

# from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# #print(f"ROC-AUC of KNN on train samples is {roc_auc_score(y_t, y_pred_prob_KNN, average='macro')}")
# print(f"Accuracy of KNN on train samples is {accuracy_score(y_t, y_pred_KNN)}")






# print(t_tr.info())
# print(t_tr.describe(include='all'))
# print(t_tr[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].isna().sum())


# col = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
#        'Transported', 'Deck',
#        'Room', 'Class', 'First_name', 'Second_name', 'Group', 'G_num']
# t_tr['Deck'] = pd.factorize(t_tr['Deck'])[0]
# t_tr['Room'] = pd.factorize(t_tr['Room'])[0]
# t_tr['Class'] = pd.factorize(t_tr['Class'])[0]


# No_money = t_tr[(t_tr['RoomService'] == 0.0) &
#            (t_tr['FoodCourt'] == 0.0) &
#            (t_tr['ShoppingMall'] == 0.0)&
#            (t_tr['Spa'] == 0.0)&
#            (t_tr['VRDeck'] == 0.0)]
    

