import panel as pn
import hvplot.pandas
import pandas as pd
import param
import panel.widgets as pnw

# Load Data
df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={"race/ethnicity": "race", "parental level of education": "parental_edu", "test preparation course": "test_prep",
                    "math score": "math_score", "reading score": "reading_score", "writing score": "writing_score"}, inplace=True)

# create a self-contained dashboard class
class InteractiveDashboard(param.Parameterized):
    
    race = param.Selector(label='Race', objects=['group A','group B','group C','group D'])
    gender = param.Selector(label='Gender', objects=['male', 'female'])
    yaxis = param.Selector(label='Y axis', objects=['math_score', 'reading_score', 'writing_score'])
    
    @param.depends('gender', 'race', 'yaxis')
    def plot_table(self):
        df_widget = pn.widgets.Tabulator(df, page_size=10, layout='fit_data_table', header_align='center')
        return df_widget
    
    def plot1(self):
        df_filtered = df[(df.gender.isin(['male', 'female'])) & (df.race.isin(['group A','group B','group C','group D']))]
        table = df_filtered.groupby(['gender', 'race'])[self.yaxis].mean().to_frame().reset_index()
        plot = table.hvplot(x='gender', y=self.yaxis, by='race', kind='bar')
        return plot
    
    def plot2(self):
        df_filtered = df[(df.gender.isin(['male', 'female'])) & (df.race.isin(['group A','group B','group C','group D']))]
        plot = df_filtered.hvplot.density(y=['math_score', 'reading_score', 'writing_score'],groupby="gender", legend='top_left')
        return plot
    
    def plot3(self):
        df_filtered = df[(df.gender == self.gender)]
        plot = df_filtered.hvplot.scatter(x='math_score', y='reading_score', by='gender', legend='top_right')
        return plot
    
dashboard = InteractiveDashboard()

# Layout using Template

template = pn.template.FastListTemplate(
    title='# My Beautiful Dashboard', 
    sidebar=[pn.Param(dashboard.param, widgets={'gender': pn.widgets.Select,'race': pn.widgets.RadioButtonGroup,'yaxis': pn.widgets.RadioButtonGroup})],
    main=[pn.Row(pn.Column(dashboard.plot_table, dashboard.plot1)),
          pn.Row(dashboard.plot2),
          pn.Row(dashboard.plot3)],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

template.servable() 
