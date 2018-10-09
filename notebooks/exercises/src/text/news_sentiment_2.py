import pandas as pd
import glob
import os
from afinn import Afinn
from ipywidgets.widgets import Accordion, Layout, Label, VBox, HTML, Dropdown, Button, Output
from IPython.core.display import display

import matplotlib # Plotting
import matplotlib.pyplot as plt # Plotting
import numpy as np # Plotting

from scipy.interpolate import spline # Smoothing matplotlib graphs

class PrimeMinisterSpeechDashboard:
    def __init__(self):
        self.afinn = Afinn(language = "da")
        self.speeches_names = [('2018 (Lars Løkke Rasmussen)', '2018'),
                         ('2017 (Lars Løkke Rasmussen)', '2017'),
                         ('2016 (Lars Løkke Rasmussen)', '2016'),
                         ('2015 (Lars Løkke Rasmussen)', '2015'),
                         ('2014 (Helle Thorning-Schmidt)', '2014'),  
                         ('2013 (Helle Thorning-Schmidt)', '2013'),  
                         ('2012 (Helle Thorning-Schmidt)', '2012'),  
                         ('2011 (Helle Thorning-Schmidt)', '2011'),  
                         ('2010 (Lars Løkke Rasmussen)', '2010'),  
                         ('2009 (Lars Løkke Rasmussen)', '2009'),  
                         ('2008 (Anders Fogh Rasmussen)', '2008'),  
                         ('2007 (Anders Fogh Rasmussen)', '2007'),  
                         ('2006 (Anders Fogh Rasmussen)', '2006'),  
                         ('2005 (Anders Fogh Rasmussen)', '2005'),  
                         ('2004 (Anders Fogh Rasmussen)', '2004'),  
                         ('2003 (Anders Fogh Rasmussen)', '2003'),  
                         ('2002 (Anders Fogh Rasmussen)', '2002'),  
                         ('2001 (Poul Nyrup Rasmussen)', '2001'),  
                         ('2000 (Poul Nyrup Rasmussen)', '2000'), 
                         ('1999 (Poul Nyrup Rasmussen)', '1999'),  
                         ('1998 (Poul Nyrup Rasmussen)', '1998'),  
                         ('1997 (Poul Nyrup Rasmussen)', '1997') 
                        ]
        self.speeches = {}
        self.speeches_sentiments = {}

        self.select = Dropdown(
            options={'2018 (Lars Løkke Rasmussen)': 0, 
                     '2017 (Lars Løkke Rasmussen)': 1, 
                     '2016 (Lars Løkke Rasmussen)': 2,  
                     '2015 (Lars Løkke Rasmussen)': 3,  
                     '2014 (Helle Thorning-Schmidt)': 4,  
                     '2013 (Helle Thorning-Schmidt)': 5,  
                     '2012 (Helle Thorning-Schmidt)': 6,  
                     '2011 (Helle Thorning-Schmidt)': 7,  
                     '2010 (Lars Løkke Rasmussen)': 8,  
                     '2009 (Lars Løkke Rasmussen)': 9,  
                     '2008 (Anders Fogh Rasmussen)': 10,  
                     '2007 (Anders Fogh Rasmussen)': 11,  
                     '2006 (Anders Fogh Rasmussen)': 12,  
                     '2005 (Anders Fogh Rasmussen)': 13,  
                     '2004 (Anders Fogh Rasmussen)': 14,  
                     '2003 (Anders Fogh Rasmussen)': 15,  
                     '2002 (Anders Fogh Rasmussen)': 16,  
                     '2001 (Poul Nyrup Rasmussen)': 17,  
                     '2000 (Poul Nyrup Rasmussen)': 18,  
                     '1999 (Poul Nyrup Rasmussen)': 19,  
                     '1998 (Poul Nyrup Rasmussen)': 20,  
                     '1997 (Poul Nyrup Rasmussen)': 21 
            },
            value=1,
            description='Vælg talen:',
            disabled=False,
            layout=Layout(width='300px'),
            style={'description_width': '130px'},
        )

        self.container = Output(
            value = "",
        )

        self.submit_button = Button(
            value=False,
            description='Indlæs talen',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Indlæs statsministerens tale og lav sentiment-analyse',
            icon=''
        )

        self.widget_box = VBox(
            (self.select, self.submit_button, self.container),
        )

        self.submit_button.on_click(self._do_sentiment_analysis)

    @property
    def start(self):
        return self.widget_box

    def load_speeches(self):
        for filepath in glob.iglob('data/statsminister/*.txt'):
            self.speeches[os.path.basename(filepath).replace(".txt","")] = [line.rstrip('\n') for line in open(filepath, mode="r", encoding="utf-8")]
            current_sentiment = 0
            for line in self.speeches[os.path.basename(filepath).replace(".txt","")]:
                current_sentiment += self.afinn.score(line)
            self.speeches_sentiments[os.path.basename(filepath).replace(".txt","")] = current_sentiment

    def _do_sentiment_analysis(self, speech_number = None):
        if speech_number:
            current_speech = speech_number
        else:
            current_speech = self.speeches_names[self.select.value][1]
        scores = []
        for i in range(len(self.speeches[current_speech])):
            scores.append(self.afinn.score(self.speeches[current_speech][i]))

        # Dataframe
        pd.set_option('display.max_colwidth', -1) # Used to display whole title (non-truncated)
        df = pd.DataFrame({"Line": self.speeches[current_speech], "Score": scores}) # Creating the data frame and populating it

        # Highlight the positive and negative sentiments
        def highlight(s):
            if s.Score > 0:
                return ['background-color: #AAFFAA']*2
            elif s.Score < 0:
                return ['background-color: #FFAAAA']*2
            else:
                return ['background-color: #FFFFFF']*2

        df = df.style.apply(highlight, axis=1)

        smoothed_scores = []
        smoothed_scores.append(scores[0])
        smoothing_constant = 0.3
        number_of_averaged_scores = 10
        for i in range(len(scores) - 1):
            s = 0
            for j in range(number_of_averaged_scores):
                if j == 0:
                    s = scores[i - j] * smoothing_constant
                elif i - j >= 0:
                    s = s + scores[i - j] * (1 - smoothing_constant / number_of_averaged_scores)        
            smoothed_scores.append(s)

        # Data for plotting
        y = np.array(smoothed_scores)
        x = np.array(range(1, len(smoothed_scores) + 1))
        x_s = np.linspace(x.min(),x.max(), 1800) #300 represents number of points to make between T.min and T.max
        y_s = spline(x, y, x_s)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x_s, y_s, color="black")
        ax.set(xlabel='Tid', ylabel='Sentiment', title="Statsministeren's tale sentiment")

        # use xnew, ynew to plot filled-color graphs
        plt.fill_between(x_s, 0, y_s, where=(y_s-1) < -1 , color='red')
        plt.fill_between(x_s, 0, y_s, where=(y_s-1) > -1 , color='green')

        ax.grid()
        plt.show()

        display(df)
        return
