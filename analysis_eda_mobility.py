import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from unidecode import unidecode
from urllib.request import urlopen
from scipy.stats import pearsonr

class DashboardEdaMobi:
    def format(self, v):
        return "{:.2f}".format(v)    
    
    # ------------------ Data quality section 
    def _s1_sec_state_filters(self, states_info, valid, k):
        
        options_state=[]
        for s in states_info:
            flag = True
            if(valid!=None):
                flag = (s in valid)
                
            if( flag ):
                name = states_info[s]['name']
                options_state.append( f"{s} - {name}" )
        state = st.selectbox( 'Choose the brazilian state:', key=k, options=options_state)
        
        return state
    
    def _s1_plot_bar_missing(self, filtered, xtitle):
        ytitle='Missing category data (%)'
        df = pd.DataFrame()
        df[xtitle] = list(filtered.keys())[:10]
        df[ytitle] = list(filtered.values())[:10]
        fig = px.bar(df, x=xtitle, y=ytitle)
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig)
    
    def _s1_plot_bar_coverage(self, filtered, xtitle):
        ytitle='Covered cities (%)'
        df = pd.DataFrame()
        df[xtitle] = list(filtered.keys())[:10]
        df[ytitle] = list(filtered.values())[:10]
        fig = px.bar(df, x=ytitle, y=xtitle, orientation='h' )
        fig.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(fig)
        
    def sec1_cols_missing(self, db, states_info):
        
        sec1 = st.container()
        with sec1: 
            st.markdown("## Data coverage")
            st.markdown("The mobility data provided by google does not cover all the cities of the brazilian states. It contains the changes in the the user usage of mobile devices in certain categories of places registered in google maps. The changes are given as percentage of difference in relation to a baseline that is considered normal according to the previous usage of the users before pandemia.")
            
            st_exist = db['state_abv'].unique()
            
            coverage_cities={}
            general_mis = {}
            general = {}
            missing = {}
            for s in st_exist:
                df = db[ db['state_abv'] == s  ]
                ns=states_info[s]['name']
                missing[s] = {}
                aux = {}
                cols = ['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','workplaces','residential']
                for c in cols:
                    aux[c] = ( len( df[ ~df[c].isna() ] ) / len(df) )*100
                    
                    val2=( len( df[ df[c].isna() ] ) / len(df) )*100
                    if(val2>0):
                        missing[s][c] = val2
                rev = dict(sorted(missing[s].items(), key=lambda item: item[1], reverse=True) )
                missing[s] = rev
                
                vals = aux.values()
                general[ns] = sum( vals )/len(vals)
                
                vals = missing[s].values()
                general_mis[ns] = sum( vals )/len(vals)
                
                total = len(states_info[s]['list'])
                covered = len(df['city'].unique())
                val = (covered/total)*100
                coverage_cities[ns] = "{:.2f}".format(val)
                
            general = dict(sorted(general.items(), key=lambda item: item[1], reverse=True) )
            general_mis = dict(sorted(general_mis.items(), key=lambda item: item[1], reverse=True) )
            coverage_cities = dict(sorted(coverage_cities.items(), key=lambda item: item[1], reverse=True) )
           
            st.markdown("Coverage of mobility change data in the state cities:")
            self._s1_plot_bar_coverage( coverage_cities, "States")
            
            st.markdown("States that have more number of missing data in the place categories:")
            self._s1_plot_bar_missing( general_mis, "States")
            
            st.markdown("Check the most absent category in each state for Mobility:")
            valid = missing.keys()
            state = self._s1_sec_state_filters(states_info, valid, 'ms1p1') 
            state = state.split(' - ')[0]
            self._s1_plot_bar_missing( missing[state], "Categories" )
    
    def _s2_category_filters(self, df):
        cols = list(df.columns)[-6:-1]
        options=[]
        for s in cols:
            name = s.replace('_',' ').capitalize()
            options.append( name )
        cats = st.selectbox( 'Choose the pace category to compare with residential:', key='ms2p1', options=options)
        
        return cats
        
    def _s2_plot_horizontal_change_boxplot(self, df, cat):
        df = df[ df['year']==2020 ]
        
        ncat = cat.replace(' ','_').lower()
        cols = ['Residential', cat]
        aux={}
        for c in cols:
            aux[c]={}
            
        for state in df['state'].unique():
            f = df[ (df['state']==state) & (~df['city'].isna()) ].fillna(0).groupby('city').mean()
            aux['Residential'][state] = f.loc[:, 'residential'].values
            aux[cat][state] = f.loc[:, ncat].values
            #print( state, len(aux[cat][state]), len(aux['Residential'][state]) )
        
        fig = go.Figure()
        colors=['#7075EB', '#3D9970']
        y = list(aux[c].keys())
        i=0
        for c in cols:  
            x=[]
            ny=[]
            for j in y:
                x += list( aux[c][j] )
                ny += [j]*len( aux[c][j] )
                
            fig.add_trace( go.Box(
                x = x,
                y = ny,
                name = c,
                marker_color = colors[i]
            ))    
            i+=1
            
        fig.update_layout( height=800, title = f"Comparison variance Residential x {cat}", yaxis_title="States", xaxis=dict(title="Mean of mobility change in the State's cities", zeroline=False), boxmode='group' )
        fig.update_traces( orientation='h' )
        
        st.plotly_chart(fig)
    
    def _s2_metric_corr_filters(self, caps):
        col1, col2, col3 = st.columns(3)
        with col1:
            capitals = list( map( lambda x: f"{x[1]} - {x[0]}",  caps.items() ) ) 
            city = st.selectbox( 'Choose a capital:', key='ms2p0', options=capitals)
        with col2:
            options=['Cases', 'Death','Cure']
            outcome = st.selectbox( 'Choose outcome:', key='ms2p3', options=options)
        with col3:
            period = st.selectbox( 'Period:', key='ms2p5', options=('Weekly', 'Monthly') )
            
        return city, outcome, period
        
    def _s2_capitals_correlation(self, df, city, outcome, fperiod):
        city = city.split(' - ')[0]
        per = fperiod[0].lower()
        f = df[ df['city']==city ]
        aux = f[ (f['outcome']==outcome) & (f['period'].str.contains(f'-{per}')) ][ ['agg_per_1000_log10', 'retail_and_recreation', 'grocery_and_pharmacy', 'parks', 'transit_stations', 'workplaces','residential'] ]
        cols = [ c.replace('_',' ').capitalize() for c in aux.columns ]
        aux.columns = cols
        aux = aux.corr()
        
        rho = aux.corr()
        pval = aux.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
        fcr = rho.round(2).astype(str) + p
        fig = px.imshow(rho, text_auto=True, width=800, height=800)
        
        st.plotly_chart(fig)
    
    def sec2_summary_states_analysis(self, db, states_info, dfts):
        
        sec1 = st.container()
        with sec1: 
            st.markdown("## Analysis comparing the mobility changes")
            
            caps = {'AC': 'Rio Branco', 'AL': 'Maceió', 'AP': 'Macapá', 'AM': 'Manaus', 'BA': 'Salvador', 'CE': 'Fortaleza', 'DF': 'Brasília', 'ES': 'Vitória', 'GO': 'Goiânia', 'MA': 'São Luís', 'MT': 'Cuiabá', 'MS': 'Campo Grande', 'MG': 'Belo Horizonte', 'PA': 'Belém', 'PB': 'João Pessoa', 'PR': 'Curitiba', 'PE': 'Recife', 'PI': 'Teresina', 'RJ': 'Rio de Janeiro', 'RN': 'Natal', 'RS': 'Porto Alegre', 'RO': 'Porto Velho', 'RR': 'Boa Vista', 'SC': 'Florianópolis', 'SP': 'São Paulo', 'SE': 'Aracaju', 'TO': 'Palmas'}
            
            st.markdown("### Correlation of latent variables with cases and outcomes for a chosen state capital")
            st.markdown("The correlation below was calculated based on the time series of the cases/outcome and the mean of mobility change observed in the chosen time period (week or month), in each city. The residential pattern was inversely related to the others place categories as expected, while among the other categories it was observed a positive correlation. The adoption of the #stayHome was clearly a rection than for prevention in Brazil, when the number of cases and deaths the city mayors took some action to restrict the people circulation, the positive correlation reflects the irresponsible measures, at a minimal signal of decrease in the number of cases most people got back to work or other places. ")
            city, outcome, period = self._s2_metric_corr_filters(caps)
            self._s2_capitals_correlation(dfts, city, outcome, period)
            
            st.markdown("### Distribution of mobility metrics along the state cities")
            st.markdown("In the plot below, it shows the chaotic distribution of the mobility changes in places fixing the residential category in comparison with the other outside places. The implementation of lockdown in Brazil severily failed, since there were strong disagreements and miscommunication among the government layers. The Plot shows that the positive change in residential was very lower than expected if a lockdown was supervised and homogenously implemented. Besides some categories such as recreation had dramatic negative changes in most states, the workplaces category had a the median value close to no change (0). Brazil is one of the most developed countries in Latin America, and suffers the burden of social inequality and it is comprehensible this result since many employers were afraid to lose their jobs if they obeyed the lockdown order. The fear of hunger in many cases won the fear of the virus.")
            cat = self._s2_category_filters(db)
            self._s2_plot_horizontal_change_boxplot(db, cat)
           
    def _s3_filters_cities(self, states_info):
        col1, col2, col3 = st.columns(3)
        with col1:
            state = self._s1_sec_state_filters( states_info, None, 'ms3p0')
        with col2:
            period = st.selectbox( 'Period:', key='ms3p2', options=('Weekly', 'Monthly') )
        with col3:
            order = st.selectbox( 'Order for top 10:', key='ms3p3', options=('Lowest to Highest', 'Highest to Lowest') )
        
        return state, period, order
    
    def _s3_plot_ranking_residential(self, df, state, period, order):
        sta = state.split(' - ')[0]
        period = period.lower().replace('ly','')
        
        df = df[ ~df['residential'].isna() ]
        df = df[ (df['state']==sta) & (df['type_period']==period) & (df['outcome']=='Cases') & ( (df['period'].str.contains('2020') ) | (df['period'].str.contains('2021')) ) ]
        df = df[ ['city', 'residential','agg_per_1000'] ].groupby('city').mean()
        
        dt={}
        aux={}
        for i in df.index:
            dt[i] = df.loc[i, 'residential']
            aux[i] = [ df.loc[i, 'residential'], df.loc[i, 'agg_per_1000'] ]
        
        flag=False
        if( order=='Highest to Lowest' ):
            flag = True
        rev = dict(sorted(dt.items(), key=lambda item: item[1], reverse=flag) )
        xs = list( rev.keys() )[:15]
        
        y1=[]
        y2=[]
        for x in xs:
            y1.append( aux[x][0] )
            y2.append( aux[x][1] )
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=xs, y=y1, name="Residential mobility change"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=xs, y=y2, name="Cases per 1000"),
            secondary_y=True,
        )

        # Add figure title
        fig.update_layout(
            title_text = f"Mobility change and cases per 1000 for top 15 cities in {state}"
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Cities")

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>(%) Residential change</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Cases per 1000</b>", secondary_y=True)
        
        st.plotly_chart(fig)
    
    def sec3_cities_change_analysis(self, states_info, dfts):
        
        sec1 = st.container()
        with sec1: 
            st.markdown("## Analysis of mobility changes in cities of a state")
            
            st.markdown("### Top 15 cities according to residential mobility in 2020-2021")
            state, period, order = self._s3_filters_cities( states_info)
            self._s3_plot_ranking_residential(dfts, state, period, order)
            
            
    def eda_UI(self, db, dfts):
        with open('filtered_data/state-city-population.json', 'r') as g:
            states_info = json.load(g)
                
        mapc = st.container()
        with mapc:
            # Data quality
            self.sec1_cols_missing(db, states_info)
            st.divider()
            
            self.sec2_summary_states_analysis(db, states_info, dfts)
            st.divider()
            
            self.sec3_cities_change_analysis( states_info, dfts)
