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
import plotly.graph_objects as go



df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={"race/ethnicity": "race", "parental level of education": "parental_edu", "test preparation course": "test_prep",
                    "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score"}, inplace=True)

male = "https://cdn-icons-png.flaticon.com/512/4537/4537047.png"
female ="https://cdn-icons-png.flaticon.com/512/4537/4537148.png"

s1=pn.Spacer(width=150)
s2=pn.Spacer(width=50)

df["Total_marks"]=((df.math_score+df.reading_score+df.writing_score)/3).round(2)

# create a self-contained dashboard class
class InteractiveDashboard(param.Parameterized):
    #parametres 
    gender = param.Selector(label='Gender', objects=['All']+list(sorted(df.gender.unique())))
    lunch=param.Selector(label="Lunch",objects=['All']+list(sorted(df.lunch.unique())))
    race = param.Selector(label='Race', objects=['All']+list(sorted(df.race.unique())))
    parental_edu=param.Selector(label="Parental_education",objects=['All']+list(sorted(df.parental_edu.unique())))
    test_prep=param.Selector(label="Test_prep",objects=['All']+list(sorted(df.test_prep.unique())))    
    gender_parent = param.Selector(objects=['parental_edu', 'gender'])


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
            marker_color='#1C4E80'
        ))
        fig.add_trace(go.Box(
            x=df_['writing_score'],
            y=df_['gender'],
            name='writing score',
            marker_color='#88C2E0'
        ))
        fig.add_trace(go.Box(
            x=df_['reading_score'],
            y=df_['gender'],
            name='reading score',
            marker_color='#DEA47E'
        ))

        fig.update_layout(
            xaxis=dict(title='Score', zeroline=False),
            boxmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=530, 
            height=430
        
        )

        fig.update_traces(orientation='h') # horizontal box plots

        return fig

    
    
    @pn.depends('gender_parent')
    def plot_barplot_stuck(self):
        colors = ['#1C4E80', '#88C2E0', '#DEA47E', '#F8F2DC']

        counts = df.groupby(['race', self.gender_parent]).size().reset_index(name='Count')
        return counts.hvplot.bar('race', 'Count', by=self.gender_parent, stacked=True, rot=90, hover_cols=['Filter'],
                                width=600, height=400, color=colors)


    def score_distribution(self):
        plot = df.hvplot.density(y=['math_score', 'reading_score', 'writing_score'],groupby="gender", legend='top_left')
        return plot

    @param.depends('gender', 'race', 'parental_edu', 'test_prep')
    def plot_pie_lunch(self):
        colors = ['#1C4E80', '#88C2E0']
        df_ =df.copy()
        sel = [self.gender,self.race, self.parental_edu, self.test_prep]
        col = ["gender","race", "parental_edu", "test_prep"]
        for i, j in zip(sel, col):
            if i != "All":
                df_ = df_[df_[j] == str(i)]
        lunch_counts =  df_ ['lunch'].value_counts()
        fig = go.Figure(data=[go.Pie(values=lunch_counts, labels=lunch_counts.index)])
        fig.update_traces(textposition='inside', textinfo='percent+label',marker=dict(colors=colors))
        fig.update_layout( width=500, height=500,plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    
    @param.depends('gender', 'race', 'parental_edu', 'lunch')
    def plot_pie_test_prep(self):
        colors = ['#DEA47E','#F8F2DC']
        df_ =df.copy()
        sel = [self.gender,self.race, self.parental_edu, self.test_prep]
        col = ["gender","race", "parental_edu", "lunch"]
        for i, j in zip(sel, col):
            if i != "All":
                df_ = df_[df_[j] == str(i)]
        lunch_counts =  df_ ['test_prep'].value_counts()
        fig = go.Figure(data=[go.Pie(values=lunch_counts, labels=lunch_counts.index)])
        fig.update_traces(textposition='inside', textinfo='percent+label',marker=dict(colors=colors))
        fig.update_layout(width=500, height=500,plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig


dashboard = InteractiveDashboard()

# Layout using Template

template = pn.template.FastListTemplate(
    title='Data Analysis', 
    sidebar=[pn.Param(dashboard.param, width=200, widgets={
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
    pn.Row(pn.Column(pn.pane.Markdown('#### Table'),dashboard.plot_table, sizing_mode='stretch_width')),
    pn.Row(pn.Column(pn.pane.Markdown('#### Lunch Distribution'),dashboard.plot_pie_lunch),s2,pn.Column(pn.pane.Markdown('#### Test Preparation Distribution'),dashboard.plot_pie_test_prep)),

    pn.Row(pn.Column(pn.pane.Markdown('#### Box Plot by gender'),dashboard.plot_box),pn.Column(pn.pane.Markdown('#### Bar plot of race by gender_parent'),dashboard.plot_barplot_stuck)),
    pn.Row(pn.Column(pn.pane.Markdown("#### Score Distribution by gender"),dashboard.score_distribution))
    ],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
