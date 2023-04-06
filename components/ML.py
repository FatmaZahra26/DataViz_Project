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
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import plotly.express as px
import plotly.subplots as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# URL of the image to download
male = "https://cdn-icons-png.flaticon.com/512/4537/4537047.png"
female="https://cdn-icons-png.flaticon.com/512/4537/4537148.png"

s1=pn.Spacer(width=150)
s2=pn.Spacer(width=50)
s3=pn.Spacer(width=150)


# Load Data
df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={"race/ethnicity": "race", "parental level of education": "parental_edu", "test preparation course": "test_prep",
                    "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score"}, inplace=True)


df["Total_marks"]=((df.math_score+df.reading_score+df.writing_score)/3).round(2)

# create a self-contained dashboard class
class InteractiveDashboard(param.Parameterized):
    #parametres 
    x_select1 = param.Selector(label='x_select 1 for scatter plot and regression model', objects=['math_score','reading_score','writing_score'])    
    x_select2 = param.Selector(label='x_select 2 for scatter plot and regression model', objects=['math_score','reading_score','writing_score'])    

    y_select = param.Selector(label='y_select for scatter plot and regression model', objects=['math_score','reading_score','writing_score'])    
    n_clusters = pn.widgets.IntSlider(name='n_clusters', start=1, end=5, value=3)


    values_moyenne = [int(df['Total_marks'].between(70, 100).sum()*100/len(df)), int(df['Total_marks'].between(50, 70).sum()*100/len(df)), int(df['Total_marks'].between(0, 50).sum()*100/len(df))]
    ind_moy1=pn.indicators.Number(name='EXCELLENCE', value=values_moyenne[0], format='{value}%',colors=[(88, 'green')],font_size='20pt',title_size='12pt')
    ind_moy2=pn.indicators.Number(name='MEDIOCRITY', value=values_moyenne[1], format='{value}%',colors=[(66, 'gold')],font_size='20pt',title_size='12pt')
    ind_moy3=pn.indicators.Number(name='FAILURE', value=values_moyenne[2],format='{value}%',colors=[(100, 'red')],font_size='20pt',title_size='12pt')

    values_gender=[round(len(df[df.gender=="male"])/len(df)*100,3),round(len(df[df.gender=="female"])/len(df)*100,2)]
    ind_gen1=pn.indicators.Number(name='Male', value=values_gender[0], format='{value}%',default_color="blue",font_size='20pt',title_size='12pt')
    ind_gen2=pn.indicators.Number(name='Female', value=values_gender[1], format='{value}%',default_color="red",font_size='20pt',title_size='12pt')

    
    @pn.depends('y_select')
    def plot_qqplot(self):
        fig, ax = plt.subplots()
        qqplot = sm.qqplot(df[self.y_select], line='45', ax=ax)
        ax.set_title(f'Q-Q Plot of {self.y_select} with Qauntile Theoriques')
        ax.set_xlabel("Qauntile Theoriques")
        ax.set_ylabel(self.y_select)
        return fig

    @pn.depends('x_select1','y_select','x_select2')
    
    def plot_scatter(self):
        df_scatter=df.hvplot.scatter(by="gender", title='Scatter', x=self.x_select1, y=self.y_select ,width=600, height=500)
        return df_scatter
    
    def heatmap(self):
        corr_matrix = df[['math_score', 'reading_score', 'writing_score']].corr()
        fig, ax = plt.subplots(figsize=(7, 7) ,facecolor='none')
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns, rotation=0)
        plt.tight_layout()

        return pn.panel(fig)
    

    @param.depends('y_select','x_select1','x_select2')
    #selection y ==> x1 , x2 fixé
    def regression_model(self):
        lr_model = LinearRegression()
        y = df[self.y_select]
        X = df[[self.x_select1,self.x_select2]]
        scaler = StandardScaler()
        #X= scaler.fit_transform(X).reshape(1,-1)
        #y= scaler.fit_transform(y).reshape(-1,1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        r2 = r2_score(y_test, y_pred) * 100
        rmse =mean_squared_error(y_test, y_pred , squared=False) 

        r2_s = pn.indicators.Number(name='R2 Score', value=r2 , format='{value:.3f}',colors=[(85, 'red'),(95,'gold'),(100,'green')],font_size='17pt',title_size='17pt')
        rmse_s = pn.indicators.Number(name='RMSE Score', value=rmse , format='{value:.3f}',colors=[(85, 'red'),(95,'gold'),(100,'green')],font_size='17pt',title_size='17pt')

        return pn.Row(s1,r2_s,s3,rmse_s)
    
    @pn.depends('y_select','x_select1', n_clusters.param.value)
    def get_clusters(self):
        scaler=StandardScaler()
        X_scaled = scaler.fit_transform(df.loc[:,['math_score', 'reading_score', 'writing_score']])
        k=self.n_clusters.param.value
        kmeans = KMeans(n_clusters=k,random_state=40).fit(X_scaled)
        
        labels = kmeans.labels_.astype(str)
        df['labels'] = labels

        centers = pd.DataFrame(kmeans.cluster_centers_, columns=["math_score", "reading_score", "writing_score"]).mean()
        centers["labels"] = centers.index.astype(str)
        return (
                    df.sort_values('labels').hvplot.scatter(
                        x=self.x_select1.param.value, 
                        y=self.y_select.param.value, 
                        c='labels', 
                        size=100, 
                        height=500
                    ) *
                    centers.hvplot.scatter(
                        x=self.x_select1.param.value, 
                        y=self.y_select.param.value, 
                        marker='x', 
                        color='black', 
                        size=400,
                        padding=0.1, 
                        line_width=5
                    )
                )
    

dashboard = InteractiveDashboard()
# Layout using Template

template = pn.template.FastListTemplate(
    title='Machine Learning', 
    sidebar=[pn.Param(dashboard.param, width=200, widgets={
        'x_select1': pn.widgets.Select,
        'x_select2': pn.widgets.Select,
        'y_select':pn.widgets.Select,
        'n_clusters': pn.widgets.IntSlider,
    })],
    sidebar_width=200,
    main=[
    pn.Row( pn.pane.PNG(male, width=60), s2, dashboard.ind_gen1, s1, pn.pane.PNG(female, width=60), dashboard.ind_gen2, align="center"),
    pn.Row(dashboard.ind_moy1, s1, dashboard.ind_moy2, s1, dashboard.ind_moy3, align="center") ,
    pn.Row(dashboard.plot_qqplot()),
    pn.Row(pn.Column(pn.pane.Markdown('## Scatter Plot'),dashboard.plot_scatter),pn.Column(pn.pane.Markdown('## HeatMap'),dashboard.heatmap())),
    pn.Row(pn.pane.Markdown('## Linear Regression Result : '),dashboard.regression_model),
    ],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
