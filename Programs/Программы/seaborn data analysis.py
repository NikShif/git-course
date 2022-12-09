
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as sts 
sns.set_style("darkgrid")

#f, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))

df = sns.load_dataset('tips')
print(df.columns)
iris = sns.load_dataset('iris')
iris = iris.drop('species', axis =1)
print(iris)
print(iris.columns)
data = pd.read_csv(r'C:\Users\shifr\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.pivot_table(index= 'size', columns =['sex', 'smoker'], values ='tip', aggfunc ='sum'))


plt.figure(figsize=(20,20))


# joint_kws = {} and marginal_kws = {} и остальные **kwargs
# либо joint_kws = {'kws_name': 'style' or number} либо joint_kws = dict(kws_name = 'style' or number)








"""________________________________  Categorical  ________________________________"""


"""_____ Count PLOT _____"""

sns.countplot(data = df, x = 'day',
              hue = 'sex',
              palette = 'magma_r').set(title = 'Days')

sns.countplot(data = df, x = 'tip',
              facecolor = (1,0,1,0),
              linewidth = 5,
              edgecolor = sns.color_palette('dark', 3)).set(title = 'Days')

"""______ BAR PLOT _____"""
# Если убрать hue, можно добиться градиента в цветовой палитре, или просто задать цвет
# Самое важно что является парметром Estimator. Именно он отвечает за то, что ообрадается на оси Y

sns.barplot(data = df, x = 'day',y = 'tip',
            order = ['Sun', 'Fri', 'Thur', 'Sat'],
            palette ='RdYlGn',
            hue = 'sex',
            hue_order = ['Male', 'Female'],
            capsize = 0.05,
            saturation = 10,
            estimator = sum).set(title = 'Bills', xlabel = 'Day', ylabel = 'Total tip sum')

"""_____ BOX PLOT _____"""
# Квантили и тд https://habr.com/ru/post/578754/

sns.boxplot(data=df, x = 'day',y = 'tip',
            hue = 'sex', showmeans = True,
            linewidth=5,
            palette = 'magma_r',
            meanprops = {'marker' : 'o',
                          'markersize' : '4',
                          'markerfacecolor' : 'white',
                          'markeredgecolor' : 'black'})

"""_____ VIOLIN PLOT _____"""
# dodge is for nesting

sns.violinplot(data = df, x = 'day', y = 'tip',
                hue = 'sex', split = True,
                inner = 'quartile',
                bw = .8, cut = 0.4,
                scale = 'width',
                ).set(title = 'Bills', xlabel = 'Day', ylabel = 'Tip dist')

"""_____ STRIP PLOT _____"""
# better to combine with violin or box plot

sns.stripplot(data=df, y='tip', x = 'day', jitter = .2,
              linewidth = 1, hue = 'sex', dodge = True, color = 'blue')

"""_____ SWARN PLOT _____"""
# same shit as strip plot< but dots are separeted 

"""_____ CAT PLOT _____"""
# Обобщение всех категориальных графиков в одной функции. Для выбора указывается kind = '*your plot style*' .
# Уникальность в том, что появляются переменные такие как row и col,
# с мопощью которых можно разделять графики доаолнително с hue

#sns.catplot(data = df, x = 'day', y = 'tip',row = 'sex', col = 'size', kind = 'strip')



"""________________________________  Distribution  ________________________________"""



"""_____ BAR PLOT _____"""
# bins можно указать интервал и тем самым отбросить выбросы и тд
# multiple shows how data wiht hue distribute. multiple : {“layer”, “dodge”, “stack”, “fill”}
# element : {“bars”, “step”, “poly”}
# Use stat = {density, probability, frequency, percent}
# Use plt.xticks(np.arange(Xmin, Xmax, Step)) to xmake proper X label

plt.xticks(np.arange(0,50,2))
sns.histplot(data = df, x = 'total_bill', hue = 'sex', 
             bins = np.arange(2,45,2),
             multiple='stack',kde = True)

sns.histplot(data = df, x = 'total_bill', 
             element = 'poly', hue = 'day')


sns.histplot(data =df, x = 'day' ,stat = 'percent', color = 'green', alpha = 0.2)

sns.histplot(data =df, x = 'tip' , y= 'total_bill')

"""_____ KDE PLOT _____"""
# kernel density https://ru.wikipedia.org/wiki/KDE
# If cumulative=True shows dist func
# thresh = n shows trash 

sns.kdeplot(data = df, x = 'total_bill', bw_adjust = 0.3,
            hue = 'sex', multiple='stack', linewidth = 1, alpha = .3,
            cumulative=True,
            palette = 'Dark2')

sns.kdeplot(data =df, x = 'tip' , y= 'total_bill', hue= 'sex',
            fill = True, levels = 4, thresh = .05)


"""_____ RUG PLOT _____"""
# Используется совместно с KDE и другми для лучшей визуализации
# height = .05 сcan be negative, but make clip_on = False

sns.kdeplot(data = df, x = 'tip' , y= 'total_bill')
sns.rugplot(data = df, x = 'tip' , y= 'total_bill', height = .05)


"""_____ ECDF PLOT _____"""
#dist func

sns.ecdfplot(data= df, x= 'tip', hue = 'sex')


"""_____ DIS PLOT _____"""
# kind = 'hist' default

sns.displot(data = df, x = 'total_bill', bins = 20,
            kde = True,
            kde_kws = dict(bw_adjust = .5), 
            rug_kws = {'height': .07, 'color': 'r'},
            stat = 'frequency',
            row = 'sex', col = 'time')



"""_____ JOINT PLOT _____"""
#Совмешение двух графиков распределения
# kind = 'scatter' default
# joint_kws = {} and marginal_kws = {} но использоваль marginal без hue
# height = 7, ratio= 3, space=0.05 общие параметры изображения
# p.plot_jointlj занит добавить еще любой plot

p = sns.jointplot(data = df, x = 'total_bill', y ='tip',
              hue = 'time',
              palette = 'BuPu',
              height = 7, ratio= 3, space=0.05)
p.plot_joint(sns.kdeplot, fill = True, alpha= .3)

"""_____ PAIR PLOT _____"""
# diag_kws = {} and plot_kws = {}
# p.off_diag() == p.map_upper() + p.map_lower()

# sns.pairplot(data = iris, diag_kind = 'kde', kind = 'kde', corner = True,
#              hue = 'species', palette = 'BuPu')

sns.pairplot(data= iris, hue = 'species', palette = 'BuPu',
              x_vars = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
              y_vars = ['sepal_length'])

p = sns.pairplot(data= iris,
                 hue = 'species', palette = 'BuPu')
p.map_upper(sns.kdeplot)






"""________________________________  Relational  ________________________________"""



"""_____ SCATTER PLOT _____"""

sns.scatterplot(data = iris, x= 'sepal_width', y='petal_length',
                #style = 'species'
                hue = 'species', s = 200,
                markers = {'setosa' : '^','virginica' : 'v', 'versicolor' : 'o'},
                palette = 'BuPu')

sns.scatterplot(data = df, x = 'total_bill', y = 'tip',
                hue = 'tip', 
                size = 'tip', sizes = (20,200),
                palette = 'BuPu')


"""_____ LINE PLOT _____"""
# Для временных рядов
# Для начала всегда лучше преобразовать df через data = data.pivot(), чтобы индеск являлся временем
# Can combine hue and size

# sns.lineplot(data = df, y='TIME', x='SOMETHING',
#               ci = True/False or ci = 'sd' or ci = n (in percent)
#               (показыввает разброс на 95 процентов уверенности для каждой точки),
#               estimator = [None, sum, median, max, min, etc.],
#               n_boot = n (определяет точность доверительного интервала)б
#               hue = 'col_name', palette = '',
#               style = 'col_name' (вместо hue; (можно указать size (sizes = (1, 20)), size_order = '')),
#               markers = True (показывает точки по которым строит))
              

"""_____ REL PLOT _____"""
# Обобщение для Relational plots
# col and rows, добавляется row_wrap & col_wrap = n (линиия из K*N плотов делится на K линиий)
     



"""________________________________  Regression  ________________________________"""

sns.regplot(data = df, x = 'total_bill', y ='tip',
            n_boot = 50)




"""________________________________  Matrix  ________________________________"""


"""_____ Heat MAP _____"""

k = df.drop(df.columns[[2,3,5,6]], axis = 1)
sns.heatmap(k.corr(), square = True,
            annot = True, fmt ='.2g', annot_kws = dict(size = 20, weight = 'bold'),
            vmin = -1, vmax = 1,
            linewidths=1,
            linecolor='black',
            cmap = 'copper').set(title = 'HEAT')








