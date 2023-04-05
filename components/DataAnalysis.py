import panel as pn
import hvplot.pandas
import pandas as pd
import plotly.express as px

import param
import panel.widgets as pnw
import seaborn as sns
from holoviews import opts
import matplotlib.pyplot as plt
import holoviews as hv
import requests
from io import BytesIO
from PIL import Image
from bokeh.io import output_notebook, show
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
import plotly.graph_objects as go

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
    gender = param.Selector(label='gender', objects=['All'] + list(sorted(df.gender.unique())))
    lunch=param.Selector(label="lunch",objects=['All'] + list(sorted(df.lunch.unique())))
    race = param.Selector(label='Race', objects=['All']+list(sorted(df.race.unique())))
    score = param.Selector(label='Score', objects=list(set(df.columns)-set(df.select_dtypes("object").columns)))
    parental_edu=param.Selector(label="Parental_education",objects=['All']+list(sorted(df.parental_edu.unique())))
    test_prep=param.Selector(label="Test_prep",objects=['All'] + list(sorted(df.test_prep.unique())))    

    gender_parent=param.Selector(label="gender_parent",objects=["parental_edu","gender"],default='gender')
    
    
    values_moyenne = [int(df['Total_marks'].between(70, 100).sum()*100/len(df)), int(df['Total_marks'].between(50, 70).sum()*100/len(df)), int(df['Total_marks'].between(0, 50).sum()*100/len(df))]
    ind_moy1=pn.indicators.Number(name='EXCELLENCE', value=values_moyenne[0], format='{value}%',colors=[(88, 'green')],font_size='20pt',title_size='12pt')
    ind_moy2=pn.indicators.Number(name='MEDIOCRITY', value=values_moyenne[1], format='{value}%',colors=[(66, 'gold')],font_size='20pt',title_size='12pt')
    ind_moy3=pn.indicators.Number(name='FAILURE', value=values_moyenne[2],format='{value}%',colors=[(100, 'red')],font_size='20pt',title_size='12pt')

    values_gender=[round(len(df[df.gender=="male"])/len(df)*100,3),round(len(df[df.gender=="female"])/len(df)*100,2)]
    ind_gen1=pn.indicators.Number(name='Male', value=values_gender[0], format='{value}%',default_color="blue",font_size='20pt',title_size='12pt')
    ind_gen2=pn.indicators.Number(name='Female', value=values_gender[1], format='{value}%',default_color="red",font_size='20pt',title_size='12pt')


    #tableau
    @param.depends('race', 'gender','lunch','parental_edu','test_prep')
    def plot_table(self):
        cm = sns.light_palette("#1C4E80", as_cmap=True)
        
        sel = [self.gender, self.lunch, self.race, self.parental_edu, self.test_prep]
        col = ["gender", "lunch", "race", "parental_edu", "test_prep"]
        df_ = df.copy()
        for i, j in zip(sel, col):
            if i != "All":
                df_ = df_[df_[j] == str(i)]
        df_=df_.iloc[:7,:]

        df_widget = pn.widgets.Tabulator(df_,
                                        header_align='center',widths={'index': '5%', 'gender': '10%', 'race': '10%', 'parental_edu': '15%', 'lunch': '10%','test_prep':'10%','math_score':'10%','reading_score':'10%','writing_score':'10%','Total_marks':'10%'},
                                        sizing_mode='stretch_width',page_size=7)

        df_widget.style.background_gradient(subset=['Total_marks'], cmap=cm, vmin=0, vmax=100)
        df_widget.style.format({'Total_marks': '{:.2f}'})
        return df_widget
    

    @param.depends("gender_parent")
    def plot_barplot_stuck(self):
        counts = df.groupby(['race', self.gender_parent]).size().reset_index(name='Count')
        return counts.hvplot.bar('race', 'Count', by=self.gender_parent, stacked=True, rot=90, hover_cols=['Filter'],width=600, height=400)
    


    def score_distribution(self):
        df_filtered = df[(df.gender.isin(['male', 'female'])) & (df.race.isin(['group A','group B','group C','group D']))]
        plot = df_filtered.hvplot.density(y=['math_score', 'reading_score', 'writing_score'],groupby="gender", legend='top_left')
        return plot

    @param.depends('gender', 'lunch', 'race', 'score', 'parental_edu', 'test_prep')
    def plot_pie(self):
        df_copy =df.copy()
        if (self.lunch!='All'):
             df_copy  =  df_copy [ df_copy ['lunch']==self.lunch]
        lunch_counts =  df_copy ['lunch'].value_counts()
        fig = px.pie(lunch_counts, values='lunch', names=lunch_counts.index)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title='Lunch Distribution')
        return fig
    
    @param.depends('race','lunch','parental_edu','test_prep')
    def plot_box(self):
        sel = [self.race,self.lunch, self.parental_edu, self.test_prep]
        col = ["race","lunch", "parental_edu", "test_prep"]
        df_=df.copy()
        for i, j in zip(sel, col):
            if i != "All":
                df_ = df_[df_[j] == str(i)]


        fig = go.Figure()
        fig.add_trace(go.Box(
            x=df_['math_score'],
            y=df_['gender'],
            name='math score',
            marker_color='#3D9970'
        ))
        fig.add_trace(go.Box(
            x=df_['writing_score'],
            y=df_['gender'],
            name='writing score',
            marker_color='#FF4136'
        ))
        fig.add_trace(go.Box(
            x=df_['reading_score'],
            y=df_['gender'],
            name='reading score',
            marker_color='#FF851B'
        ))

        fig.update_layout(
            xaxis=dict(title='Score', zeroline=False),
            boxmode='group'
        )

        fig.update_traces(orientation='h') # horizontal box plots

        return fig 
   

dashboard = InteractiveDashboard()

# Layout using Template

template = pn.template.FastListTemplate(

    title='Data Analysis', 
    sidebar=[
    pn.Param(dashboard.param, width=200, widgets={
        'gender': pn.widgets.Select,
        'lunch':pn.widgets.Select ,
        'race': pn.widgets.Select,
        'score': pn.widgets.Select,
        'test_prep': pn.widgets.Select,
        'parental_edu': pn.widgets.Select,
    })],

    sidebar_width=200,

    main=[
    pn.Row( pn.pane.PNG(male, width=60), s2, dashboard.ind_gen1, s1, pn.pane.PNG(female, width=60), dashboard.ind_gen2, align="center"),
    pn.Row(dashboard.ind_moy1, s1, dashboard.ind_moy2, s1, dashboard.ind_moy3, align="center") ,
    pn.Row(dashboard.plot_table, sizing_mode='stretch_width'),
    pn.Row(pn.Column(pn.pane.Markdown('## Barplot '),dashboard.plot_barplot_stuck)),
    pn.Row(dashboard.score_distribution()),
    pn.Row(dashboard.plot_box()),
    pn.Row(dashboard.plot_pie()),
    ],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
