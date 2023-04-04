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



# URL of the image to download
male = "https://cdn-icons-png.flaticon.com/512/4537/4537047.png"
female="https://cdn-icons-png.flaticon.com/512/4537/4537148.png"

s1=pn.Spacer(width=25)
s2=pn.Spacer(width=2)


# Load Data
df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={"race/ethnicity": "race", "parental level of education": "parental_edu", "test preparation course": "test_prep",
                    "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score"}, inplace=True)

df["moyenne"]=((df.math_score+df.reading_score+df.writing_score)/3).round(2)



# create a self-contained dashboard class
class InteractiveDashboard(param.Parameterized):
    #parametres 
    gender = param.Selector(label='gender', objects=list(sorted(df.gender.unique())))
    lunch=param.Selector(label="lunch",objects=list(sorted(df.lunch.unique())))
    race = param.Selector(label='Race', objects=['All']+list(sorted(df.race.unique())))
    score = param.Selector(label='Score', objects=list(set(df.columns)-set(df.select_dtypes("object").columns)))
    parental_edu=param.Selector(label="Parental_education",objects=['All']+list(sorted(df.parental_edu.unique())))
    test_prep=param.Selector(label="Test_prep",objects=['All']+list(sorted(df.test_prep.unique())))    

    gender_parent=param.Selector(label="gender_parent",objects=["parental_edu","gender"],default='gender')
    
    
    values_moyenne = [int(df['moyenne'].between(70, 100).sum()*100/len(df)), int(df['moyenne'].between(50, 70).sum()*100/len(df)), int(df['moyenne'].between(0, 50).sum()*100/len(df))]
    ind_moy1=pn.indicators.Number(name='EXCELLENCE', value=values_moyenne[0], format='{value}%',colors=[(88, 'green')],font_size='20pt',title_size='12pt')
    ind_moy2=pn.indicators.Number(name='MEDIOCRITY', value=values_moyenne[1], format='{value}%',colors=[(66, 'gold')],font_size='20pt',title_size='12pt')
    ind_moy3=pn.indicators.Number(name='FAILURE', value=values_moyenne[2],format='{value}%',colors=[(100, 'red')],font_size='20pt',title_size='12pt')

    values_gender=[len(df[df.gender=="male"])*100/len(df),len(df[df.gender=="female"])*100/len(df)]
    ind_gen1=pn.indicators.Number(name='Male', value=values_gender[0], format='{value}%',default_color="blue",font_size='20pt',title_size='12pt')
    ind_gen2=pn.indicators.Number(name='Female', value=values_moyenne[1], format='{value}%',default_color="red",font_size='20pt',title_size='12pt')


    #tableau
    @pn.depends('race', 'gender', 'score','lunch','parental_edu','test_prep')
    def plot_table(self):
        cm = sns.light_palette("#1C4E80", as_cmap=True)
        df_widget = pn.widgets.Tabulator(df, header_align='center', layout='fit_data', page_size=5)
        df_widget.style.background_gradient(subset=['moyenne'], cmap=cm, vmin=0, vmax=100)
        df_widget.style.format({'moyenne': '{:.2f}'})
        return df_widget
    
    def plot_scatter(self):
        df_scatter=df.hvplot.scatter(by=['gender'], title='Scatter', x='writing_score', y=['math_score'],width=500, height=400)
        return df_scatter

    @param.depends("gender_parent")
    def plot_barplot_stuck(self):
        counts = df.groupby(['race', self.gender_parent]).size().reset_index(name='Count')
        return counts.hvplot.bar('race', 'Count', by=self.gender_parent, stacked=True, rot=90, hover_cols=['Filter'],width=600, height=400)
    

    def plot_box(self):
        df_box = df.hvplot.box(y=['math_score', 'reading_score', 'writing_score'], 
                legend=False, value_label='Score Box Plot', invert=True,width=600, height=300)

        return df_box 

    def plot_swarm(self):
        sns.set_style('whitegrid')
        f, ax = plt.subplots(1,1,figsize= (15,5))
        df_swarm = sns.swarmplot(x ='math_score', y='gender', data = df , palette = 'Set1')
        plt.title('Math Score Distribution by Gender')
        plt.xlabel('Math Score')
        plt.ylabel('Gender')
        plt.tight_layout()
        return f
    

    def heatmap(self):
        corr_matrix = df[['math_score', 'reading_score', 'writing_score']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns, rotation=0)
        plt.tight_layout()

        return pn.panel(fig)

""" 
    def plot2(self):
        df_filtered = df[(df.gender.isin(['male', 'female'])) & (df.race.isin(['group A','group B','group C','group D']))]
        plot = df_filtered.hvplot.density(y=['math_score', 'reading_score', 'writing_score'],groupby="gender", legend='top_left')
        return plot
    
    def plot3(self):
        df_filtered = df[(df.gender == self.gender)]
        plot = df_filtered.hvplot.scatter(x='math_score', y='reading_score', by='gender', legend='top_right')
        return plot
    """
dashboard = InteractiveDashboard()

# Layout using Template

template = pn.template.FastListTemplate(
    title='# My Beautiful Dashboard', 
    sidebar=[pn.Param(dashboard.param,width=200,widgets={
        'gender': pn.widgets.CheckButtonGroup(name='gender',Width=0.1,button_type='success',options=list(sorted(df.gender.unique()))),
        'lunch':pn.widgets.CheckButtonGroup(name='lunch',Width=0.1,button_type='success',options=list(sorted(df.lunch.unique()))),
        'race': pn.widgets.Select,
        'score': pn.widgets.Select,
        'test_prep': pn.widgets.Select,
        'parental_edu': pn.widgets.Select,
        })],
    sidebar_width=200,
    main=[
    pn.Row( pn.pane.PNG(male,width=60),s2,dashboard.ind_gen1,s1,pn.pane.PNG(female,width=60),s2,dashboard.ind_gen2,s1,dashboard.ind_moy1,s1,dashboard.ind_moy2,s1,dashboard.ind_moy3),    pn.Row(dashboard.plot_table),
    pn.Row(dashboard.plot_barplot_stuck,dashboard.plot_scatter),
    pn.Row(dashboard.plot_box, dashboard.heatmap()),
    pn.Row(dashboard.plot_swarm()),
   ],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
