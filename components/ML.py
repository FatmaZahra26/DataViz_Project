import panel as pn
import hvplot.pandas
import pandas as pd
import param
import panel.widgets as pnw
import seaborn as sns
from holoviews import opts
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# URL of the image to download
male = "https://cdn-icons-png.flaticon.com/512/4537/4537047.png"
female="https://cdn-icons-png.flaticon.com/512/4537/4537148.png"

s1=pn.Spacer(width=10)
s2=pn.Spacer(width=0)


# Load Data
df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={"race/ethnicity": "race", "parental level of education": "parental_edu", "test preparation course": "test_prep",
                    "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score"}, inplace=True)


df["Total_marks"]=((df.math_score+df.reading_score+df.writing_score)/3).round(2)

# create a self-contained dashboard class
class InteractiveDashboard(param.Parameterized):
    #parametres 
    y_select = param.Selector(label='y_select', objects=list(set(df.columns)-set(df.select_dtypes("object").columns)))

    values_moyenne = [int(df['Total_marks'].between(70, 100).sum()*100/len(df)), int(df['Total_marks'].between(50, 70).sum()*100/len(df)), int(df['Total_marks'].between(0, 50).sum()*100/len(df))]
    ind_moy1=pn.indicators.Number(name='EXCELLENCE', value=values_moyenne[0], format='{value}%',colors=[(88, 'green')],font_size='20pt',title_size='12pt')
    ind_moy2=pn.indicators.Number(name='MEDIOCRITY', value=values_moyenne[1], format='{value}%',colors=[(66, 'gold')],font_size='20pt',title_size='12pt')
    ind_moy3=pn.indicators.Number(name='FAILURE', value=values_moyenne[2],format='{value}%',colors=[(100, 'red')],font_size='20pt',title_size='12pt')

    values_gender=[len(df[df.gender=="male"])*100/len(df),len(df[df.gender=="female"])*100/len(df)]
    ind_gen1=pn.indicators.Number(name='Male', value=values_gender[0], format='{value}%',default_color="blue",font_size='20pt',title_size='12pt')
    ind_gen2=pn.indicators.Number(name='Female', value=values_moyenne[1], format='{value}%',default_color="red",font_size='20pt',title_size='12pt')

    @pn.depends("y_select")
    
    def plot_scatter(self):
        df_scatter=df.hvplot.scatter(by=['gender'],x='Total_marks' , y=self.y_select, title='Scatter', width=700, height=500)
        return df_scatter
    
    def regression_model(self):
        y = df['Total_marks']
        X = df.drop(['Total_marks', 'gender', 'race', 'parental_edu', 'test_prep','lunch'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        #y_pred = lr_model.predict(X_test)
        r2 = lr_model.score(X_test, y_test) * 100

        return pn.indicators.Number(name='R2 score', value=r2, format='{value:.3f}',font_size='20pt',title_size='12pt')


dashboard = InteractiveDashboard()

# Layout using Template

template = pn.template.FastListTemplate(
    title='Machine Learning', 
    sidebar=[pn.Param(dashboard.param, width=200, widgets={
        'y_select': pn.widgets.Select,
          })],
    sidebar_width=200,
    main=[
    pn.Row( pn.pane.PNG(male, width=60), s2, dashboard.ind_gen1, s1, pn.pane.PNG(female, width=60), dashboard.ind_gen2, align="center"),
    pn.Row(dashboard.ind_moy1, s1, dashboard.ind_moy2, s1, dashboard.ind_moy3, align="center") ,

    pn.Row(dashboard.plot_scatter()),
    pn.Row(dashboard.regression_model()),
    ],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
