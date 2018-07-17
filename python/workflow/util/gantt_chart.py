#!/usr/bin/python

"""
Simple function to plot a plotly gantt chart for finishing plan
"""
import plotly.plotly as py
import plotly.figure_factory as ff

def finishing_gantt(outfile):
    """
    Function to make a weekly gantt chart for finishing purposes
    :return:
    """
    df = [dict(Task='Spatial clustering and stress inversion Rotokawa (and Southern Ngatamariki?)',
               Start='2018-07-16', Finish='2018-07-23'),
          dict(Task='Remake Ngatamariki figures with injectivity included',
               Start='2018-07-16', Finish='2018-07-23'),
          dict(Task='Apply for new visa through end of study',
               Start='2018-07-16', Finish='2018-07-23'),
          dict(Task='Run final hypoDD cc calculations and polarity clustering on Rotokawa data (on servers)',
               Start='2018-07-23', Finish='2018-07-30'),
          dict(Task='Finalize consensus/composite polarity clusters for both fields (custom HASH input)',
               Start='2018-07-30', Finish='2018-08-20'),
          dict(Task='Finalize FEHM model for NM08 stimulation',
               Start='2018-08-13', Finish='2018-08-27'),
          dict(Task='Write plotting functionality for FEHM stress output and comparison to Arnold/Townend stress inversions',
               Start='2018-08-27', Finish='2018-09-03'),
          dict(Task='Create and run models for stimulation of NM10 and Ngatamariki plant startup',
               Start='2018-09-03', Finish='2018-09-17'),
          dict(Task='Write chapter: Spatial and temporal variation in focal mechanisms at Ngatamariki/Rotokawa',
               Start='2018-09-17', Finish='2018-10-15'),
          dict(Task='Write chapter: FM stress inversion compared to coupled thermo-hydro-mechanical finite element models',
               Start='2018-10-15', Finish='2018-11-12'),
          dict(Task='Finish writing catalog chapter: Add Rotokawa catalog',
               Start='2018-11-12', Finish='2018-12-10'),
          # dict(Task='',
          #      Start='2018-10-08', Finish='2018-10-15'),
          # dict(Task='',
          #      Start='2018-10-15', Finish='2018-10-22'),
          # dict(Task='',
          #      Start='2018-10-22', Finish='2018-10-29'),
          # dict(Task='',
          #      Start='2018-10-29', Finish='2018-11-05'),
          # dict(Task='',
          #      Start='2018-11-05', Finish='2018-11-12'),
          # dict(Task='',
          #      Start='2018-11-12', Finish='2018-11-19'),
          # dict(Task='',
          #      Start='2018-11-19', Finish='2018-11-26'),
          # dict(Task='',
          #      Start='2018-11-26', Finish='2018-12-03'),
          # dict(Task='',
          #      Start='2018-12-03', Finish='2018-12-10'),
          # dict(Task='',
          #      Start='2018-12-10', Finish='2018-12-17'),
          # dict(Task='',
          #      Start='2018-12-17', Finish='2018-12-24'),
          ]
    fig = ff.create_gantt(df)
    fig['layout'].update(autosize=False, width=1600, height=800,
                         margin=dict(l=600))
    py.plot(fig, filename=outfile, title='Chet Hopp: PhD Finishing Timeline')
    return