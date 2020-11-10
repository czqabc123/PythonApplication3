import numpy as np
import matplotlib.pyplot as plt	
import pandas as pd
df = pd.read_csv("heart.csv")
df.head()
#import pandas_profiling
#profile = pandas_profiling.ProfileReport(df)
#profile.to_file('profile.html')
import seaborn as sns
df.corr()#相关性分析
#plt.figure(figsize = (12,12))
#sns.heatmap(df.corr(),vmax =3,center=0,annot=True,linewidths= 5,cbar_kws={"shrink":.5},fmt='.1f',square=True)
#plt.tight_layout()
#plt.show()
#sns.pairplot(df)
#plt.show()
#sns.distplot(df['age'])
#plt.show()
#sns.countplot(x='target',data=df,palette='bwr')
#plt.show()
#plt.xlabel('Sex(0=female,1=male)')
#plt.show()
#pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
#plt.xlabel('Age')
#plt.ylabel('Frequency')
#plt.savefig('age-heartdisease')
#plt.show()
#sns.boxplot(x=df['target'],y=df['age'])
#plt.show()
#sns.violinplot(x=df['target'],y=df['age'])
#plt.show()
#plt.scatter(x=df.age[df.target ==1],y=df.thalach[(df.target==1)],c="red")
#plt.scatter(x=df.age[df.target ==0],y=df.thalach[(df.target==0)],c="blue")
#plt.legend(["Disease","Not Disease"])
#plt.xlabel("Age")
#plt.ylabel("Max Rate")
#plt.show()
df.columns = ['age','sex','chest_pain_type','resting_blood_pressure','cholesterol','fasting_blood_sugar','rest_ecg','max_heart_rate','exercise_induced_angina','st_depression','st_slope','num_major_vessels','thalassemia','target']
#df.head()
df['sex'][df['sex']==0]='female'
df['sex'][df['sex']==1]='male'
df['chest_pain_type'][df['chest_pain_type']==0]='typical_angina'
df['chest_pain_type'][df['chest_pain_type']==1]='atypical_angina'
df['chest_pain_type'][df['chest_pain_type']==2]='non_angina pain'
df['chest_pain_type'][df['chest_pain_type']==3]='asymptomatic'

df['fasting_blood_sugar'][df['fasting_blood_sugar']==0]='lower than 120'
df['fasting_blood_sugar'][df['fasting_blood_sugar']==1]='more than 120'

df['rest_ecg'][df['rest_ecg']==0]='normal'
df['rest_ecg'][df['rest_ecg']==1]='ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg']==2]='left ventricular hypertrophy'

df['exercise_induced_angina'][df['exercise_induced_angina']==0]='no'
df['exercise_induced_angina'][df['exercise_induced_angina']==1]='yes'

df['st_slope'][df['st_slope']==0]='upsloping'
df['st_slope'][df['st_slope']==1]='flat'
df['st_slope'][df['st_slope']==2]='downsloping'

df['thalassemia'][df['thalassemia']==0]='unknown'
df['thalassemia'][df['thalassemia']==1]='normal'
df['thalassemia'][df['thalassemia']==2]='fixed defect'
df['thalassemia'][df['thalassemia']==3]='reserable defect'
df.head()
df = pd.get_dummies(df) 
df.to_csv('process_heart.csv',index=False)
from pdpbox import pdp,get_dataset,info_plots
#fig,axes,summary_df=info_plots.target_plot(df =df,feature='sex_male',feature_name='gender',target=['target'])
#_=axes['bar_ax'].set_xticklabels(['Female','Male'])
#plt.show()
#summary_df
#fig,axes,summary_df=info_plots.target_plot(df =df,feature='num_major_vessels',feature_name='num_vessels',target=['target'])
#_=axes['bar_ax'].set_xticklabels(['Female','Male'])
#fig,axes,summary_df=info_plots.target_plot(df =df,feature='age',feature_name='age',target=['target'])
#_=axes['bar_ax'].set_xticklabels(['Female','Male'])
print(df)
X = df.drop('target',axis=1)#吧除了target之外的列作为特征
print(X.shape)
Y = df['target']#取出所有属于target下面的数据作为y
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_tests = train_test_split(X,Y,test_size=0.2,random_state = 10)
print(X_train.shape)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=5)
model.fit(X_train,y_train)
len(model.estimators_)
estimator = model.estimators_[7]#指定索引为7的决策树
print(estimator)
#将特征值转化为字符串
feature_names = X_train.columns
y_train_str = y_train.astype('str')
y_train_str[y_train_str=='0']='no disease'
y_train_str[y_train_str=='1']='disease'
y_trian_str=y_train_str.values
from sklearn.tree import export_graphviz
export_graphviz(estimator,out_file='tree.dot',feature_names=feature_names,class_names=['no disease','disease'],rounded = True,proportion=True,label='root',precision=2,filled=True)
#下面的dot是linux下面才有的  是为了把上面生成的文本tree.dot转化为png
#from subprocess import call
#call(['dot','-Tpng','tree.dot','-o','tree.png','-Gdpi=600'])
#由于上一步做不到，本步骤忽略  其实不影响什么，只是树不能可视化了而已
#from IPython.display import Image
#Image(filename ='tree.png')
import eli5
eli5.show_weights(estimator,feature_names=feature_names.to_list())
model.feature_importances_
print('特征排序：')
feature_names = X_test.columns
feature_importances = model.feature_importances_
indices=np.argsort(feature_importances)[::-1]
plt.figure(figsize=(16,8))
plt.title("Feature Importance")
plt.bar(range(len(feature_importances)),feature_importances[indices],color='b')
plt.xticks(range(len(feature_importances)),np.array(feature_names)[indices],color='b',rotation=90)
plt.show()