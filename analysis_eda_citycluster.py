import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import json
import os
import networkx as nx

from scipy.spatial import distance

from streamlit_agraph import agraph, Node, Edge, Config

class DashboardClustering:
    def format(self, v):
        return "{:.2f}".format(v)    
    
    def sec1_build_network(self, dfts, x):
        cities = list(dfts['city'].unique())
        
        sec1 = st.container()
        with sec1:
            st.markdown("### Cities grouped by behavior patterns")
            st.write("Network formed by the distance among the numerical features of the cities. Blue nodes are the capitals and the green ones represent the other cities. The numerical features were extracted by taking the mean of ages per 1000 (for Cases and Deaths) or residential change percentage (for mobility change) along the weeks of 2020 and 2021. The columns of this dataset are the periods of time (each week) and the rows are the cities.")
            
            col1, col2 = st.columns(2)
            with col1:
                metric = st.selectbox( 'Metric to group cities:', key='cs1p0', options = ('Deaths','Cases', 'Residential mobility change') )
            with col2:
                dcut = st.slider('Max distance to form network', 0, 20, 10)
                
            if(metric=='Residential mobility change'):
                metric='mobility'
                
            m = distance.cdist( x[metric], x[metric], 'euclidean' )
            g = nx.Graph()
            
            hubs={}
            i=0
            for x in m:
                c1=cities[i]
                hubs[c1]=0
                j=0
                for y in x:
                    if(i<j):
                        c2=cities[j]
                        
                        if(y<dcut):
                            g.add_edge(c1, c2)
                            hubs[c1]+=1
                    j+=1
                i+=1
            
            coms = nx.connected_components(g)
            dt={}
            for c in coms:
                for el in c:
                    if( not el in dt):
                        dt[el]=0
                    dt[el]+=1
                    
            gcoms = dict(sorted(dt.items(), key=lambda item: item[1], reverse=True) )
            hubs = dict(sorted(hubs.items(), key=lambda item: item[1], reverse=True) )
            stars = list(hubs.keys())[:20]
            with st.expander("Top 20 hubs list"):
                for s in stars:
                    st.markdown("- "+s+" - Degree: "+str(hubs[s])+" - Participated in "+str(gcoms[s])+" groups" )
            
            nset=set()
            eset=set()
            nodes=[]
            edges=[]
            
            caps = {'AC': 'Rio Branco', 'AL': 'Maceió', 'AP': 'Macapá', 'AM': 'Manaus', 'BA': 'Salvador', 'CE': 'Fortaleza', 'DF': 'Brasília', 'ES': 'Vitória', 'GO': 'Goiânia', 'MA': 'São Luís', 'MT': 'Cuiabá', 'MS': 'Campo Grande', 'MG': 'Belo Horizonte', 'PA': 'Belém', 'PB': 'João Pessoa', 'PR': 'Curitiba', 'PE': 'Recife', 'PI': 'Teresina', 'RJ': 'Rio de Janeiro', 'RN': 'Natal', 'RS': 'Porto Alegre', 'RO': 'Porto Velho', 'RR': 'Boa Vista', 'SC': 'Florianópolis', 'SP': 'São Paulo', 'SE': 'Aracaju', 'TO': 'Palmas'}
            capitals = list(caps.values())
            
            hubs={}
            i=0
            for x in m:
                c1=cities[i]
                hubs[c1]=0
                j=0
                for y in x:
                    if(i<j):
                        c2=cities[j]
                        
                        if(y<dcut):
                            ide=f"{c1}-{c2}"
                            if( not ide in nset):
                                nset.add(ide)
                                edges.append( Edge(source=c1,  label=f"d: {format(y)}", target=c2 ) )
                            
                            size=20
                            color='#37AA20'
                            if(c1 in capitals):
                                size=25
                                color='#3374FF'
                                
                            if(not c1 in nset):
                                nset.add(c1)
                                nodes.append( Node(id=c1, label=c1, size=size, color=color) )
                            
                            size=20
                            color='#37AA20'
                            if(c2 in capitals):
                                size=25
                                color='#3374FF'
                                
                            if(not c2 in nset):
                                nset.add(c2)
                                nodes.append( Node(id=c2, label=c2, size=size, color=color) )
                    j+=1
                i+=1
            
            config = Config(width=750, height=950, directed=True,  physics=True,  hierarchical=False )

            return_value = agraph(nodes=nodes, edges=edges, config=config)
            
    def eda_UI(self, dfts):
        with open('x_clustering.json') as g:
            x = json.load(g)
            
        mapc = st.container()
        with mapc:
            st.markdown("## City clustering according to time series evolution of cases or mobility change")
            self.sec1_build_network(dfts, x)
            st.divider()
