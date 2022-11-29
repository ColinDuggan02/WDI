#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Basic Packages
from __future__ import division
import os
from datetime import datetime

# Web & file access
import requests
import io

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

get_ipython().run_line_magic('pylab', '--no-import-all')
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_context("talk")

import plotly.express as px
import plotly.graph_objects as go

from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
# Next line can import all of plotnine, but may overwrite things? Better import each function/object you need
#from plotnine import *

# Data
import pandas as pd
import numpy as np
from pandas_datareader import data, wb

# GIS & maps
import geopandas as gpd
gp = gpd
import georasters as gr
import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc
import textwrap

# Data Munging
from itertools import product, combinations
import difflib
import pycountry
import geocoder
from geonamescache.mappers import country
mapper = country(from_key='name', to_key='iso3')
mapper2 = country(from_key='iso3', to_key='iso')
mapper3 = country(from_key='iso3', to_key='name')

# Regressions & Stats
from scipy.stats import norm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer, LineLocation

# Paths
pathout = './data/'

if not os.path.exists(pathout):
    os.mkdir(pathout)
    
pathgraphs = './graphs/'
if not os.path.exists(pathgraphs):
    os.mkdir(pathgraphs)
#code:
wdi_indicators_patents = ['IP.PAT.NRES', 'IP.PAT.RESD']
#https://data.worldbank.org/indicator/IP.PAT.NRES
#https://data.worldbank.org/indicator/IP.PAT.RESD
wdi = wb.download(
    indicator=wdi_indicators_patents, country=list_of_countries_ISO_A2_codes, start=start_year, end=end_year)


# <div class="alert alert-block alert-warning">
#     <b>Exercise 1:</b> Get WDI data on <b>patent applications by residents and non-residents in each country</b>. Create a new variable that shows the total patents for each country.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Exercise 2:</b> Using the <code>my_xy_plot</code> function plot the relation between <b>GDP per capita</b> and <b>total patents</b> in the years 1990, 1995, 2000, 2010, 2020.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Exercise 3:</b> Using the <code>my_xy_line_plot</code> function plot the evolution of <b>GDP per capita</b> and <b>total patents</b> by income groups and regions (separate figures).
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Exercise 4:</b> Plot the relation between patenting activity by <b>residents and non-residents</b> in the year 2015. Make sure to show the 45 degree line so you can see how similar they are.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Exercise 5:</b> Create a static and a dynamic map for patenting activity in the year 2015 across the world. 
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Exercise 6:</b> Explore the relation between economic development as measured by Log[GDP per capita] and patenting activity. Show the relation for residents, non-residents, and total, all in one nice looking table. Also, produce a few nice looking figures.
# </div>

# In[7]:


from pandas_datareader import wb, data
import pandas as pd


# In[8]:


wbcountries = wb.get_countries()
wbcountries = wbcountries[wbcountries['region'] != "Aggregates"]
wbcountries = wbcountries[wbcountries['iso3c'].notnull()]


wbcountries


# In[9]:


# non_residents
non_residents = wb.download(indicator="IP.PAT.NRES", country=wbcountries.iso2c.values, start=1950, end=2020)
non_residents = non_residents[non_residents['IP.PAT.NRES'].notnull()]
non_residents = non_residents.reset_index()
residents = wb.download(indicator="IP.PAT.RESD", country=wbcountries.iso2c.values, start=1950, end=2020)
residents = residents[residents['IP.PAT.RESD'].notnull()]
residents = residents.reset_index()
residents.head()


# In[10]:


# patents
patents = residents.merge(non_residents, on=['country', 'year'])
patents = patents.reset_index()
patents['total_patents'] = patents['IP.PAT.RESD']+ patents['IP.PAT.NRES']

patents.head()


# In[11]:


df = wbcountries.merge(patents, left_on='name', right_on='country')
df.head()


# In[26]:


wdi_indicators = ['NY.GDP.PCAP.PP.KD', 'NY.GDP.PCAP.KD', 'SL.GDP.PCAP.EM.KD', 'SP.POP.GROW', 'SP.POP.TOTL', 'SP.DYN.WFRT', 'SP.DYN.TFRT.IN']
wdi = wb.download(indicator = wdi_indicators, 
                 country=wbcountries.iso2c.values, start=1950, end=2020 )
wdi


# In[32]:


wdi = wdi.reset_index()


# In[34]:


df = df.merge(wdi, left_on=['country', 'year'], right_on=['country', 'year'])
df


# In[35]:


def my_xy_plot(dfin, 
               x='SP.POP.GROW', 
               y='ln_gdp_pc', 
               labelvar='iso3c', 
               dx=0.006125, 
               dy=0.006125, 
               xlogscale=False, 
               ylogscale=False,
               xlabel='Growth Rate of Population', 
               ylabel='Log[Income per capita in ' +  str(2020) + ']',
               labels=False,
               xpct = False,
               ypct = False,
               OLS=False,
               OLSlinelabel='OLS',
               ssline=False,
               sslinelabel='45 Degree Line',
               filename='income-pop-growth.pdf',
               hue='region',
               hue_order=['East Asia & Pacific', 'Europe & Central Asia',
                          'Latin America & Caribbean ', 'Middle East & North Africa',
                          'North America', 'South Asia', 'Sub-Saharan Africa '],
               style='incomeLevel', 
               style_order=['High Income', 'Upper Middle Income', 'Lower Middle Income', 'Low Income'],
               palette=None,
               size=None,
               sizes=None,
               legend_fontsize=10,
               label_font_size=12,
               save=True):
    '''
    Plot the association between x and var in dataframe using labelvar for labels.
    '''
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_context("talk")
    df = dfin.copy()
    df = df.dropna(subset=[x, y]).reset_index(drop=True)
    # Plot
    k = 0
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, data=df, ax=ax, 
                    hue=hue,
                    hue_order=hue_order,
                    #hue='incomeLevel',
                    #hue_order=['High Income', 'Upper Middle Income', 'Lower Middle Income', 'Low Income'],
                    #hue_order=['East Asia & Pacific', 'Europe & Central Asia',
                    #           'Latin America & Caribbean ', 'Middle East & North Africa',
                    #           'North America', 'South Asia', 'Sub-Saharan Africa '],
                    alpha=1, 
                    style=style, 
                    style_order=style_order,
                    palette=palette,
                    size=size,
                    sizes=sizes,
                    #palette=sns.color_palette("Blues_r", df[hue].unique().shape[0]+6)[:df[hue].unique().shape[0]*2:2],
                )
    if OLS:
        sns.regplot(x=x, y=y, data=df, ax=ax, label=OLSlinelabel, scatter=False)
    if ssline:
        ax.plot([df[x].min()*.99, df[x].max()*1.01], [df[x].min()*.99, df[x].max()*1.01], c='r', label=sslinelabel)
    if labels:
        movex = df[x].mean() * dx
        movey = df[y].mean() * dy
        for line in range(0,df.shape[0]):
            ax.text(df[x][line]+movex, df[y][line]+movey, df[labelvar][line], horizontalalignment='left', fontsize=label_font_size, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xpct:
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks)
    if ypct:
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
    if ylogscale:
        ax.set(yscale="log")
    if xlogscale:
        ax.set(xscale="log")
    handles, labels = ax.get_legend_handles_labels()
    handles = np.array(handles)
    labels = np.array(labels)
    handles = list(handles[(labels!=hue) & (labels!=style) & (labels!=size)])
    labels = list(labels[(labels!=hue) & (labels!=style) & (labels!=size)])
    ax.legend(handles=handles, labels=labels, fontsize=legend_fontsize)
    if save:
        plt.savefig(pathgraphs + filename, dpi=300, bbox_inches='tight')
    return fig


# In[36]:


df['year'] = df.year.astype(int)
df['gdp_pc'] = df['NY.GDP.PCAP.PP.KD']
df['ln_gdp_pc'] = df['NY.GDP.PCAP.PP.KD'].apply(np.log)
df['ln_pop'] = df['SP.POP.TOTL'].apply(np.log)

df['name'] = df.name.str.strip()
df['incomeLevel'] = df['incomeLevel'].str.title()
df.loc[df.iso3c=='VEN', 'incomeLevel'] = 'Upper Middle Income'


# In[37]:


years = [1990, 1995, 2000, 2010, 2020]
df_gdp_pc_patents = df[df['year'].isin(years)].dropna(subset=['ln_gdp_pc','ln_pop'])
df_gdp_pc_patents = df_gdp_pc_patents.sort_values(by='region').reset_index()
df_gdp_pc_patents['total_patents'] = df_gdp_pc_patents['total_patents'].apply(np.log)
df_gdp_pc_patents


# In[38]:


g = my_xy_plot(df_gdp_pc_patents, 
               x='ln_gdp_pc', 
               y='total_patents', 
               xlabel='Log[GDP per capita]', 
               ylabel='Total Patents', 
               OLS=True, 
               labels=True, 
#                ylogscale = True,
               #size="ln_pop", 
               #sizes=(10, 400), 
               filename='ln-gdp-pc-total-patents.pdf')


# In[39]:


df.ln_gdp_pc.mean()


# In[42]:


dfin = df.dropna(subset=['ln_gdp_pc','ln_pop'])
dfin = dfin.sort_values(by='region').reset_index()
dfin['total_patents'] = dfin['total_patents'].apply(np.log)
dfin


# In[45]:


def my_xy_line_plot(dfin, 
               x='SP.POP.GROW', 
               y='ln_gdp_pc', 
               labelvar='iso3c', 
               dx=0.006125, 
               dy=0.006125, 
               xlogscale=False, 
               ylogscale=False,
               xlabel='Growth Rate of Population', 
               ylabel='Log[Income per capita in ' +  str(2020) + ']',
               labels=False,
               xpct = False,
               ypct = False,
               OLS=False,
               OLSlinelabel='OLS',
               ssline=False,
               sslinelabel='45 Degree Line',
               filename='income-pop-growth.pdf',
               hue='region',
               hue_order=['East Asia & Pacific', 'Europe & Central Asia',
                          'Latin America & Caribbean ', 'Middle East & North Africa',
                          'North America', 'South Asia', 'Sub-Saharan Africa '],
               style='incomeLevel', 
               style_order=['High Income', 'Upper Middle Income', 'Lower Middle Income', 'Low Income'],
               palette=None,
               fontsize=10,
               save=True):
    '''
    Plot the association between x and var in dataframe using labelvar for labels. 
    '''
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_context("talk")
    df = dfin.copy()
    df = df.dropna(subset=[x, y]).reset_index(drop=True)
    # Plot
    k = 0
    fig, ax = plt.subplots()
    sns.lineplot(x=x, y=y, data=df, ax=ax, 
                    hue=hue,
                    hue_order=hue_order,
                    alpha=1, 
                    style=style, 
                    style_order=style_order,
                    palette=palette,
                )
    if OLS:
        sns.regplot(x=x, y=y, data=df, ax=ax, label=OLSlinelabel, scatter=False)
    if ssline:
        ax.plot([df[x].min()*.99, df[x].max()*1.01], [df[x].min()*.99, df[x].max()*1.01], c='r', label=sslinelabel)
    if labels:
        movex = df[x].mean() * dx
        movey = df[y].mean() * dy
        for line in range(0,df.shape[0]):
            ax.text(df[x][line]+movex, df[y][line]+movey, df[labelvar][line], horizontalalignment='left', fontsize=12, color='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xpct:
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        ax.xaxis.set_major_formatter(xticks)
    if ypct:
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
    if ylogscale:
        ax.set(yscale="log")
    if xlogscale:
        ax.set(xscale="log")
    handles, labels = ax.get_legend_handles_labels()
    handles = np.array(handles)
    labels = np.array(labels)
    handles = list(handles[(labels!='region') & (labels!='incomeLevel')])
    labels = list(labels[(labels!='region') & (labels!='incomeLevel')])
    ax.legend(handles=handles, labels=labels, fontsize=fontsize)
    if save:
        plt.savefig(pathgraphs + filename, dpi=300, bbox_inches='tight')
    return fig


# In[48]:


palette=sns.color_palette("Blues_r", df['incomeLevel'].unique().shape[0]+6)[:df['incomeLevel'].unique().shape[0]*2:2]
fig = my_xy_line_plot(dfin, 
                x='total_patents', 
                y='ln_gdp_pc', 
                xlabel='Total Patents',
                ylabel='Log[GDP per capita]',
                filename='ln-gdp-pc-ls-total-patents.pdf',
                hue='incomeLevel',
                hue_order=['High Income', 'Upper Middle Income', 'Lower Middle Income', 'Low Income'],
                palette=palette,
                OLS=False, 
                labels=False,
                save=True,
                     ylogscale = True,)


# In[50]:


fig = my_xy_line_plot(dfin, 
                      x='total_patents', 
                      y='gdp_pc', 
                      xlabel='Total Patents',
                      ylabel='GDP per capita',
                      ylogscale=True,
                      filename='ln-gdp-pc-regions-TS.pdf',
                      style='region',
                      style_order=['East Asia & Pacific', 'Europe & Central Asia',
                                   'Latin America & Caribbean ', 'Middle East & North Africa',
                                   'North America', 'South Asia', 'Sub-Saharan Africa '],
                      #palette=palette,
                      OLS=False, 
                      labels=False,
                      save=True)


# In[51]:


df_2015 = df[df['year'] == 2015]
df_2015


# In[52]:


g = my_xy_plot(df_2015, 
               x='IP.PAT.NRES', 
               y='IP.PAT.RESD', 
               xlabel='Residential Patents', 
               ylabel='Non-Residential Patents', 
               OLS=True, 
               labels=True, 
#                ylogscale = True,
#                xlogscale = True,
               #size="ln_pop", 
               #sizes=(10, 400), 
               filename='res-vs-non-res-patents.pdf')


# In[53]:


headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}

url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip'
r = requests.get(url, headers=headers)
countries = gp.read_file(io.BytesIO(r.content))


# In[54]:


df_2015 = countries.merge(df_2015, left_on='ADM0_A3', right_on='iso3c')
fig, ax = plt.subplots(figsize=(15,10))
df_2015.plot(column='total_patents', ax=ax, cmap='Reds')
ax.set_title("2015 Patents", fontdict={'fontsize':34})


# In[55]:


scheme = mc.Quantiles(df_2015['total_patents'], k=5)
classifier = mc.Quantiles.make(k=5, rolling=True)
df_2015['total_patents_q'] = classifier(df_2015['total_patents'])
df_2015['total_patents_qc'] = df_2015['total_patents_q'].apply(lambda x: scheme.get_legend_classes()[x].replace('[   ', '[').replace('( ', '('))


# In[58]:


fig = px.choropleth(df_2015.sort_values('total_patents', ascending=True), 
                    locations="iso3c",
                    color="total_patents_qc",
                    hover_name='name',
                    hover_data=['iso3c', 'ln_pop'],
                    labels={
                        "total_patents": "Total Patents (" + str(2020) + ")",
                    },
                    color_discrete_sequence=px.colors.sequential.Reds,
                    height=600, 
                    width=1000,
                   )
fig.show()


# In[60]:


fig = go.Figure(data=go.Choropleth(
    locations = df_2015['iso3c'],
    z = df_2015['total_patents'],
    text = df_2015['name'],
    colorscale = 'Blues',
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix = '',
    colorbar_title = 'Total Patents',
    )                  
)
fig.update_layout(
    autosize=False,
    width=800,
    height=400,
    margin=dict(
        l=5,
        r=5,
        b=10,
        t=10,
        pad=1
    ),
    paper_bgcolor="LightSteelBlue",
)
fig.show()


# In[61]:


filtered_df = df[['year', 'name', 'ln_gdp_pc', 'total_patents', 'IP.PAT.RESD','IP.PAT.NRES', 'region', 'incomeLevel', 'iso3c']]
filtered_df


# In[62]:


print(filtered_df.ln_gdp_pc.min())
print(filtered_df.ln_gdp_pc.max())


# In[63]:


plt.plot(filtered_df['ln_gdp_pc'], filtered_df['total_patents'])
plt.xlabel("Log [GDP Per Capita]")
plt.ylabel("Total Patents")
plt.title("Total Patents vs Log [GDP Per Capita]")
plt.show()


# In[64]:


plt.plot(filtered_df['ln_gdp_pc'], filtered_df['IP.PAT.RESD'])
plt.xlabel("Log [GDP Per Capita]")
plt.ylabel("Residential Patents")
plt.title("Residential Patents vs Log [GDP Per Capita]")
plt.show()


# [<center><img src="https://github.com/measuring-culture/Expanding-Measurement-Culture-Facebook-JRSI/blob/main/pics/SMUlogowWordmarkRB.jpg?raw=true" width="250"></center>](http://omerozak.com)
