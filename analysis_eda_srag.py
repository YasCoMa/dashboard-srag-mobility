import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from unidecode import unidecode
from urllib.request import urlopen
from scipy.stats import pearsonr

class DashboardEdaSrag:
    def format(self, v):
        return "{:.2f}".format(v)    
    
    # ------------------ Data quality section 
    def _s1_sec_state_filters(self, states_info, valid, k):
        
        options_state=[]
        for s in states_info:
            if(s in valid):
                name = states_info[s]['name']
                options_state.append( f"{s} - {name}" )
        state = st.selectbox( 'Choose the brazilian state:', key=k, options=options_state)
        
        return state
    
    def _s1_plot_bar_missing(self, filtered, xtitle):
        ytitle='Missing data (%)'
        df = pd.DataFrame()
        df[xtitle] = list(filtered.keys())[:10]
        df[ytitle] = list(filtered.values())[:10]
        fig = px.bar(df, x=xtitle, y=ytitle)
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig)
    
    def _s1_plot_bar_wrongcity(self, filtered):
        xtitle="Cities"
        ytitle="Number of Records"
        df = pd.DataFrame()
        df[xtitle] = list(filtered.keys())
        df[ytitle] = list(filtered.values())
        fig = px.bar(df, x=xtitle, y=ytitle)
        #fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig)
        
    def sec1_cols_missing(self, db, states_info):
        
        sec1 = st.container()
        with sec1: 
            st.markdown("## Data quality")
            st.markdown("The original dataset was filtered according to get only the records with positive response in the rt-pcr test for SARS2. Many cities, specifically with small population size, are underepresented. This tool DOES NOT reflect the real and complete severity and loss caused by this virus in the Brazilian population.")
            
            st_exist = db['state_notification'].unique()
            general = {}
            missing = {}
            for s in st_exist:
                df = db[ db['state_notification'] == s  ]
                ns=states_info[s]['name']
                missing[s] = {}
                cols = db.columns
                for c in cols:
                    filters=[" ( df[c]=='ignored' ) "]
                    if(c=='pregnancy'):
                        filters.append(" ( df['gender']=='female' ) ")
                    filters = ' & '.join(filters)
                    val=( len( df[ eval(filters) ] ) / len(df) )*100
                    if(val>0):
                        missing[s][c] = val
                rev = dict(sorted(missing[s].items(), key=lambda item: item[1], reverse=True) )
                missing[s] = rev
                
                vals = missing[s].values()
                general[ns] = sum( vals )/len(vals)
            general = dict(sorted(general.items(), key=lambda item: item[1], reverse=True) )
           
            st.markdown("States that have more number of missing data in columns:")
            self._s1_plot_bar_missing( general, "States")
           
            st.markdown("Check the most absent information in each state for SRAG:")
            valid = missing.keys()
            state = self._s1_sec_state_filters(states_info, valid, 's1p1') 
            state = state.split(' - ')[0]
            self._s1_plot_bar_missing( missing[state], "Columns" )
            
            st.markdown("In srag dataset, the missing data portion is very low. From the columns that have more absent information, besides race and pregnancy are great indicators for directed epidemiological studies, in the case of pregnancy, this field was ignored in general by 20% or less of female records. In general, missing data in race column corresponds to a relevant portion that varies from 52% in Distrito Federal, maintains in in 25-32 in most of the states, and has a minimum of 3.83% in Acre.")
            
            st.markdown("The city names were normalized to uppercase and without accents in the original dataset. These city names were then mapped to the brazilian city names according to the [2022 census table](https://www.ibge.gov.br/estatisticas/sociais/trabalho/22827-censo-demografico-2022.html?edicao=35938). In all data, considering the years of 2020, 2021 and 2022, only 12 cities could not be mapped due to typos in the city names. The amoun of lost records by each wrong city is showed below.")
            with open('../filtered_data/wrong-city-info.json', 'r') as g:
                cinfo = json.load(g)
            dt={}
            for c in cinfo:
                dt[c] = sum(cinfo[c].values())
            self._s1_plot_bar_wrongcity(dt)
            
    # ------------------ Data summary entire state section 
    def _s2_sec_state_filters(self, states_info):
        
        options_state=[]
        for s in states_info:
            name = states_info[s]['name']
            options_state.append( f"{s} - {name}" )
        state = st.selectbox( 'Choose the brazilian state:', key='s2p1', options=options_state)
        
        options_year = ['All']
        for i in range(2020, 2023):
            options_year.append( str(i) )
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox( 'Choose the Year:', options_year)
        with col2:
            metric = st.selectbox( 'Choose the Scope:', ('All cases', 'Cure', 'Death') )
        
        return state, year, metric
    
    def _s2_plot_map_state_cities(self, dbf, nmetric, nyear, state, states_info):
        abvst = state.split(' - ')[0]
        state_name = state.split(' - ')[1]
        lat = states_info[abvst]['lat']
        lon = states_info[abvst]['lon']
        state_code = states_info[abvst]['code']
        
        with urlopen( f'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-{state_code}-mun.json') as response:
            geo_json = json.load(response)
        
        #f = db[ (db['state_notification']==abvst) & (db['year']==int(year) ) & (db['outcome']==metric) ][ ['city_notification', 'date_notification'] ].groupby(by='city_notification').count()
        f = dbf[ ['city_notification', 'date_notification'] ].groupby(by='city_notification').count()
        filtered = pd.DataFrame()
        filtered['name'] = list(f.index)
        filtered['metric'] = (f['date_notification']+1).values
        filtered['log_metric'] = np.log( f['date_notification']+1 ).values
        
        map_ui = px.choropleth_mapbox(filtered,
         geojson = geo_json,
         locations="name",
         featureidkey = 'properties.name',
         color = "log_metric",
         hover_name = 'name',
         hover_data = ['metric'],
         title = f"COVID-19 - {state_name} - {nmetric} {nyear}",
         color_continuous_scale="Viridis",
         mapbox_style = "carto-positron", 
         center = {"lat": lat, "lon": lon},
         zoom = 5,
         opacity = 0.9, )
        map_ui.update_geos(fitbounds = "locations", visible = False)
        
        st.plotly_chart(map_ui)
    
    def _s2_capitals_correlation(self, df, city):
        city = city.split(' - ')[0]
        
        aux=pd.DataFrame()
        aux['Cases']=df[ df['city_notification']==city ].groupby('week')['race'].count()
        aux['Deaths']=df[ (df['city_notification']==city) & (df['outcome']=='Death') ].groupby('week')['race'].count()
        aux['Cure']=df[ (df['city_notification']==city) & (df['outcome']=='Cure') ].groupby('week')['race'].count()
        aux['Delay outcome']=df[ df['city_notification']==city ].groupby('week')['delay_outcome'].mean()
        aux['Delay notification']=df[ df['city_notification']==city ].groupby('week')['delay_report'].mean()
        aux['Displacement for medical care']=df[ (df['city_notification']==city) & (df['displacement_for_medical_care']=='yes') ].groupby('week')['race'].count()
        
        rho = aux.corr()
        pval = aux.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
        fcr = rho.round(2).astype(str) + p
        fig = px.imshow(rho, text_auto=True)
        
        st.plotly_chart(fig)
    
    def _s2_capitals_summary(self, df, capitals, states_info):
        x=[]
        y=[]
        z=[]
        for city in capitals:
            state = city.split(' - ')[1]
            city = city.split(' - ')[0]
            pop = list( filter( lambda x: x['name']==city, states_info[state]['list'] ) )[0]['pop']
            
            temp = df[ df['city_notification']==city ].groupby('week')['race'].count()
            norm = (sum(temp.values)*100)/pop
            y += [norm]
            x += [city]
            z += ['Cases']
            
            temp = df[ (df['city_notification']==city) & (df['outcome']=='Death') ].groupby('week')['race'].count()
            norm = (sum(temp.values)*100)/pop
            y += [norm]
            x += [city]
            z += ['Deaths']
            
            temp = df[ (df['city_notification']==city) & (df['outcome']=='Cure') ].groupby('week')['race'].count()
            norm = (sum(temp.values)*100)/pop
            y += [norm]
            x += [city]
            z += ['Cure']
            
            temp = df[ (df['city_notification']==city) & (df['displacement_for_medical_care']=='yes') ].groupby('week')['race'].count()
            norm = (sum(temp.values)*100)/pop
            y += [norm]
            x += [city]
            z += ['Displacement for medical care']
            
        dff=pd.DataFrame()
        dff['City'] = x
        dff['Value per 100 people'] = y
        dff['Metric'] = z
        fig = px.line(dff, x='City', y='Value per 100 people', color='Metric', width=1000)
        
        st.plotly_chart(fig)
    
    def sec2_cases_death_state_summary(self, db, states_info):
        # select case/death
        # select state, id=abv, name=name
        
        sec1 = st.container()
        with sec1:
            caps = {'AC': 'Rio Branco', 'AL': 'Maceió', 'AP': 'Macapá', 'AM': 'Manaus', 'BA': 'Salvador', 'CE': 'Fortaleza', 'DF': 'Brasília', 'ES': 'Vitória', 'GO': 'Goiânia', 'MA': 'São Luís', 'MT': 'Cuiabá', 'MS': 'Campo Grande', 'MG': 'Belo Horizonte', 'PA': 'Belém', 'PB': 'João Pessoa', 'PR': 'Curitiba', 'PE': 'Recife', 'PI': 'Teresina', 'RJ': 'Rio de Janeiro', 'RN': 'Natal', 'RS': 'Porto Alegre', 'RO': 'Porto Velho', 'RR': 'Boa Vista', 'SC': 'Florianópolis', 'SP': 'São Paulo', 'SE': 'Aracaju', 'TO': 'Palmas'}
            capitals = list( map( lambda x: f"{x[1]} - {x[0]}",  caps.items() ) ) 
            """
            st.markdown("### Correlation of latent variables with cases and outcomes for a chosen state capital")
            city = st.selectbox( 'Choose a brazilian capital:', key='s2p0', options=capitals)
            self._s2_capitals_correlation(db, city)
            """
            st.markdown("### Total summary of main metrics in the Brazilian state capitals")
            self._s2_capitals_summary(db, capitals, states_info)
            
            st.markdown("### Situation by year in the geographical region of the chosen state")
            state, year, metric = self._s2_sec_state_filters(states_info)
            
            # ----- Map of cities in the state -------
            try:
                abvst = state.split(' - ')[0]
                nmetric=metric
                nyear=''
                filters=[ "(db['state_notification']==abvst)" ]
                if( year!='All'):
                    nyear = f"in {nyear}"
                    filters.append( "(db['year']==int(year) )" )
                if( metric!='All cases'):
                    nmetric+=" cases"
                    filters.append( "(db['outcome']==metric )" )
                filters = ' & '.join(filters)
                dbf=db[ eval(filters) ]
                self._s2_plot_map_state_cities(dbf, nmetric, nyear, state, states_info)
            except:
                pass
            
    # ------------------ Data analysis by time period 
    def _s3_sec_state_period_filters(self, states_info):
        
        options_state=[]
        for s in states_info:
            name = states_info[s]['name']
            options_state.append( f"{s} - {name}" )
        state = st.selectbox( 'Choose the brazilian state:', key='s3p1', options=options_state)
        
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox( 'Period:', key='s3p2', options=('Weekly', 'Monthly'))
        with col2:
            metric = st.selectbox( 'Choose the Scope:', key='s3p3', options=('All cases', 'Cure', 'Death') )
        
        return state, period, metric
    
    def _s3_plot_spread_disease_rate(self, db, states_info):
        options_state=[]
        for s in states_info:
            name = states_info[s]['name']
            options_state.append( f"{s} - {name}" )
        values_list = st.multiselect( f'Choose the states to compare (up to 4):', options_state, default = ['AP - Amapá', 'RJ - Rio de Janeiro', 'SP - São Paulo', 'CE - Ceará'], max_selections=4)
        
        periods=set()
        dt={}
        for v in values_list:
            state = v.split(' - ')[0]
            name = v.split(' - ')[1]
            dt[name] = {}
            cts=set()
            f = db[ db['state_notification']==state ]
            for i in f.index:
                w = f.loc[i, 'week']
                    
                ct = f.loc[i, 'city_notification']
                if(not ct in cts):
                    cts.add(ct)
                    periods.add(w)
                    if(not w in dt[name]):
                        dt[name][w]=0
                    dt[name][w]+=1
            #print( name, ':', len(cts), '/', len(f['city_notification'].unique()))
        
        possible_xs=[]
        for i in ['2020','2021','2022']:
            for j in range(1, 60):
                j=str(j)
                if(len(j)==1):
                    j='0'+j
                possible_xs.append(i+'-w'+j)
                
        st.markdown("Brazil had three great waves of COVID-19 cases, in each wave states located in distinct brazilian regions obtained notifications belonging to new cities. Rio de Janeiro and São Paulo (both in Southeast) are two of the biggest air traffic hubs and it is expected that new cases were rapidly spread along the first half of 2020. Ceará (Northeast) receives a lot of people most for tourism, and even then the notifications followed a similar timeline pattern in relation to SP and RJ.")
        fig = go.Figure()
        for pr in dt:
            aux={}
            for p in possible_xs:
                if(p in periods):
                    aux[p]=0
                    if(p in dt[pr]):
                        aux[p] = dt[pr][p]
            fig.add_trace(go.Scatter(x=list(aux.keys()), y = list(aux.values()), mode='lines+markers', name=pr))
        fig.update_layout( title = f"Chronological disease spread in the states", xaxis_title = f"Period in weeks", yaxis_title="Number of new cities")
        st.plotly_chart(fig)
                    
    
    def _s3_col_filters(self, dbf, col, title):
        dbf = dbf[ ~dbf[col].isna() ]
        col_values = ['All']
        col_values += list( filter( lambda x: x!='ignored', dbf[col].unique() ) )
        values_list = st.multiselect( f'Choose {title}:', col_values, default = ['All'])
        return values_list
   
    def _s3_plot_grouped_info(self, dbf, nmetric, fperiod, col, title, fvalues):
        if(col!='risk_factor_binary'):
            dbf = dbf[ ~dbf[col].isna() ][ [fperiod, col] ]
            
        aux={}
        periods=set()
        for i in dbf.index:
            per = dbf.loc[i, fperiod]
            
            ncol=col
            if(col=='risk_factor_binary'):
                ncol='risk_factor'
                
            preg = dbf.loc[i, ncol]
            if(col=='risk_factor'):
                els=preg.split(';')
                
                for preg in els:
                    flag=True
                    if( not 'All' in fvalues):
                        flag = (preg in fvalues)
                    if(flag):    
                        if(not preg in aux):
                            aux[preg]={}
                        
                        periods.add(per)
                        if(not per in aux[preg]):
                            aux[preg][per]=0
                        aux[preg][per]+=1    
            elif(col=='risk_factor_binary'):
                preg = str(preg)
                clas='Without risk factor'
                if(preg!='nan'):
                    clas='With risk factor'
                preg=clas
                if(not preg in aux):
                    aux[preg]={}
                
                periods.add(per)
                if(not per in aux[preg]):
                    aux[preg][per]=0
                aux[preg][per]+=1  
            else:
                flag=True
                if( not 'All' in fvalues):
                    flag = (preg in fvalues)
                    
                if(flag):    
                    if(not preg in aux):
                        aux[preg]={}
                    
                    periods.add(per)
                    if(not per in aux[preg]):
                        aux[preg][per]=0
                    aux[preg][per]+=1 
                
        chosen=list(aux.keys())
        if(col=='risk_factor'):
            gen={}
            for pr in aux:
                gen[pr]=sum(aux[pr].values())
                
            rev = dict(sorted(gen.items(), key=lambda item: item[1], reverse=True) )
            chosen = list(rev.keys())[:10]
        
        possible_xs=[]
        if(fperiod=='week'):
            for i in ['2020','2021','2022']:
                for j in range(1, 60):
                    j=str(j)
                    if(len(j)==1):
                        j='0'+j
                    possible_xs.append(i+'-w'+j)
                    
        if(fperiod=='month'):
            for i in ['2020','2021','2022']:
                for j in range(1, 12):
                    j=str(j)
                    if(len(j)==1):
                        j='0'+j
                    possible_xs.append(i+'-m'+j)
                
        if( len(chosen) > 0):
            fig = go.Figure()
            for pr in chosen:
                dt={}
                for p in possible_xs:
                    if(p in periods):
                        dt[p]=0
                        if(p in aux[pr]):
                            dt[p] = aux[pr][p]
                fig.add_trace(go.Scatter(x=list(dt.keys()), y = list(dt.values()), mode='lines+markers', name=pr))
            fig.update_layout( title = f"{title} - {nmetric}", xaxis_title = f"Period in {fperiod}s", yaxis_title="Count")
            #fig = px.line(df, x=f'Period ({fperiod}s)', y=f'{nmetric} Records', color='Pregnancy Information' )
            #fig.update_layout(yaxis_range=[0, 100])
            
            st.plotly_chart(fig)
        else:
            st.markdown(f"{title} - {nmetric}: There is no data to plot")
    
    def sec3_grouped_info_time_summary(self, db, states_info):
        # select case/death
        # select state, id=abv, name=name
        
        sec1 = st.container()
        with sec1:
            st.markdown("### Descriptive analysis along time periods")
            
            # ----- descriptive analysis using time dimension -------
            self._s3_plot_spread_disease_rate(db, states_info)
            
            state, period, metric = self._s3_sec_state_period_filters(states_info)
            
            abvst = state.split(' - ')[0]
            nmetric=metric
            filters=[ "(db['state_notification']==abvst)" ]
            if( metric!='All cases'):
                nmetric+=" cases"
                filters.append( "(db['outcome']==metric )" )
            filters = ' & '.join(filters)
            dbf=db[ eval(filters) ]
                
            fperiod='week'
            if(period=='Monthly'):
                fperiod='month'
            
            dff = dbf[ (dbf['pregnancy']!='ignored') & (dbf['pregnancy']!='no') & (dbf['pregnancy']!='Pregnancy age ignored') ].sort_values(by='date_notification')
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'pregnancy', 'Pregnancy Stages', ['All'])
            
            dff = dbf.sort_values(by='date_notification')
            
            fvaluesg = self._s3_col_filters( dff, 'gender', 'Gender')
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'gender', 'Gender', fvaluesg)
            
            fvaluesa = self._s3_col_filters( dff, 'age_group', 'Age groups')
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'age_group', 'Age groups', fvaluesa)
            
            fvaluesr = self._s3_col_filters( dff, 'race', 'Race')
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'race', 'Race', fvaluesr)
            
            fvaluesri = self._s3_col_filters( dff, 'risk_factor', 'Risk factors')
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'risk_factor', 'Risk factors', fvaluesri)
            
            self._s3_plot_grouped_info(dff, nmetric, fperiod, 'risk_factor_binary', 'Existence of Risk factors', ['All'])
    
    # ------------------ Data analysis by cities 
    def _s4_sec_state_metric_filters(self, states_info):
        
        options_state=[]
        for s in states_info:
            name = states_info[s]['name']
            options_state.append( f"{s} - {name}" )
        
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox( 'Choose the brazilian state:', key='s4p1', options=options_state)
        with col2:
            metric = st.selectbox( 'Choose the Scope:', key='s4p2', options=('All cases', 'Cure', 'Death') )
        
        return state, metric
    
    def _s4_plot_grouped_city_info(self, dbf, nmetric, col, title):
        if(col!='risk_factor_binary'):
            dbf = dbf[ ~dbf[col].isna() ][ ['city_notification', col] ]
            
        aux={}
        for i in dbf.index:
            per = dbf.loc[i, 'city_notification']
            
            ncol=col
            if(col=='risk_factor_binary'):
                ncol='risk_factor'
                
            preg = str(dbf.loc[i, ncol])
            if(col=='risk_factor'):
                els=preg.split(';')
                
                for preg in els:
                    flag=True
                    #if( not 'All' in fvalues):
                    #    flag = (preg in fvalues)
                    if(flag):    
                        if(not preg in aux):
                            aux[preg]={}
                        
                        if(not per in aux[preg]):
                            aux[preg][per]=0
                        aux[preg][per]+=1    
            elif(col=='risk_factor_binary'):
                preg = str(preg)
                clas='Without risk factor'
                if(preg!='nan'):
                    clas='With risk factor'
                preg=clas
                if(not preg in aux):
                    aux[preg]={}
                
                if(not per in aux[preg]):
                    aux[preg][per]=0
                aux[preg][per]+=1  
            else:
                flag=True
                #if( not 'All' in fvalues):
                #    flag = (preg in fvalues)
                    
                if(flag):    
                    if(not preg in aux):
                        aux[preg]={}
                    
                    if(not per in aux[preg]):
                        aux[preg][per]=0
                    aux[preg][per]+=1 
                
        #chosen=list(aux.keys())
        xs=set()
        gen={}
        for pr in aux:
            gen[pr]=sum(aux[pr].values())
            rev = dict(sorted(aux[pr].items(), key=lambda item: item[1], reverse=True) )
            top15 = list(rev.keys())[1:16] # removing the capital
            ranked={}
            for k in top15: # up to 15 cities
                xs.add(k)
                ranked[k]=rev[k]
            aux[pr]=ranked   
            
        rev = dict(sorted(gen.items(), key=lambda item: item[1], reverse=True) )
        chosen = list(rev.keys())[:10] # up to 10 legend colors
                
        if( len(chosen) > 0):
            fig = go.Figure()
            for pr in chosen:
                dt={}
                
                for p in xs:
                    dt[p]=0
                    if (p in aux[pr]):
                        dt[p] = aux[pr][p]
                fig.add_trace(go.Scatter(x=list(xs), y = list(dt.values()), mode='lines+markers', name=pr))
            fig.update_layout( title = f"{title} - {nmetric}", xaxis_title = f"Top 15 cities", yaxis_title="Count")
            #fig = px.line(df, x=f'Period ({fperiod}s)', y=f'{nmetric} Records', color='Pregnancy Information' )
            #fig.update_layout(yaxis_range=[0, 100])
            
            st.plotly_chart(fig)
        else:
            st.markdown(f"{title} - {nmetric}: There is no data to plot")
    
    def sec4_grouped_info_city_summary(self, db, states_info):
        # select case/death
        # select state, id=abv, name=name
        
        sec1 = st.container()
        with sec1:
            state, metric = self._s4_sec_state_metric_filters(states_info)
            
            st.markdown("### Descriptive analysis along top ranked cities in the chosen state")
            
            # ----- descriptive analysis using city as base -------
            abvst = state.split(' - ')[0]
            nmetric=metric
            filters=[ "(db['state_notification']==abvst)" ]
            if( metric!='All cases'):
                nmetric+=" cases"
                filters.append( "(db['outcome']==metric )" )
            filters = ' & '.join(filters)
            dbf=db[ eval(filters) ]
            
            dff = dbf.sort_values(by='date_notification')
            
            self._s4_plot_grouped_city_info(dff, nmetric, 'risk_factor', 'Risk factors')
            
            self._s4_plot_grouped_city_info(dff, nmetric, 'risk_factor_binary', 'Existence of Risk factors')
            
            self._s4_plot_grouped_city_info(dff, nmetric, 'delay_report', 'Time took between first symptoms and notification')
            
            self._s4_plot_grouped_city_info(dff, nmetric, 'delay_outcome', 'Time took between notification and outcome')
            
            self._s4_plot_grouped_city_info(dff, nmetric, 'displacement_for_medical_care', 'People that seek medical assistance in another city')
            
    def eda_UI(self, db):
        with open('../filtered_data/state-city-population.json', 'r') as g:
            states_info = json.load(g)
                
        mapc = st.container()
        with mapc:
            # Data quality
            self.sec1_cols_missing(db, states_info)
            st.divider()
            
            # Data summary by state
            st.markdown("## Data exploration by state")
            self.sec2_cases_death_state_summary(db, states_info)
            self.sec3_grouped_info_time_summary(db, states_info)
            st.divider()
            
            # Date summary in cities
            self.sec4_grouped_info_city_summary(db, states_info)
