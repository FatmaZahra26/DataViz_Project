import panel as pn
import hvplot.pandas
import pandas as pd
import param
import panel.widgets as pnw
import seaborn as sns

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
    

    #gender_parent=param.Selector(label="gender_parent",objects=["parental_edu","gender"],default='gender')

    #tableau
    @pn.depends('race', 'gender', 'score')
    def plot_table(self):
        cm = sns.light_palette("#1C4E80", as_cmap=True)
        df_widget = pn.widgets.Tabulator(df, header_align='center', layout='fit_data', page_size=5)
        df_widget.style.background_gradient(subset=['moyenne'], cmap=cm, vmin=0, vmax=100)
        df_widget.style.format({'moyenne': '{:.2f}'})
        return df_widget
    
    def plot_scatter(self):
        df.hvplot(by=[''], kind='scatter', title='Penguins Scatter', x='bill_length_mm', y=['bill_depth_mm'])

    #@param.depends("gender_parent")
    #def plot_barplot_stuck(self):
    #    counts = df.groupby(['race', self.gender_parent]).size().reset_index(name='Count')
    #    return counts.hvplot.bar('race', 'Count', by=self.gender_parent, stacked=True, rot=90, hover_cols=['Filter'])



   
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
    sidebar=[pn.Param(dashboard.param,widgets={
        'gender': pn.widgets.CheckButtonGroup(name='gender',Width=0.2,button_type='success',options=list(sorted(df.gender.unique()))),
        'lunch':pn.widgets.CheckButtonGroup(name='lunch',Width=0.2,button_type='success',options=list(sorted(df.lunch.unique()))),
        'race': pn.widgets.Select,
        'score': pn.widgets.Select,
        'test_prep': pn.widgets.Select,
        'parental_edu': pn.widgets.Select,
        })],
    main=[pn.Row(dashboard.plot_table)],
          #pn.Row(dashboard.plot_barplot_stuck)],
    accent_base_color="#1C4E80",
    header_background="#1C4E80",
    background_color="#EDF0F3",
)

template.servable() 
