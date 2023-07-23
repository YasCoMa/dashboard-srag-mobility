import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance

# ---------- Prepare input matrix ---------------
def make_input():
    if(not os.path.isfile('x_clustering.json')):
        ts = pd.read_csv('../filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
        possible_xs=[]
        for i in ['2020','2021','2022']:
            for j in range(1, 60):
                j=str(j)
                if(len(j)==1):
                    j='0'+j
                possible_xs.append(i+'-w'+j)
        
        outs=['Cases','Deaths']
        existing_xs = set(ts['period'].unique())
        newcols = {}
        for p in possible_xs:
            if(p in existing_xs):
                newcols[p] = 0
        
        # each line is a city, goal is clustering cities using cases, and after using mobi residential, and calculate clusters overlapping in the two scenarios
        cities = list(ts['city'].unique())
        outcomes = ['Cases','Deaths', 'mobility']
        
        xs = {}
        for o in outcomes:
            xs[o]=[]
        
        ind=1
        for c in cities:
            for o in outcomes:
                #o='Cases'
                #for o in outcomes:
                cond = "(ts['city']==c) & (ts['outcome']=='Cases')"
                col='residential'
                if( o!='mobility' ):
                    cond = " (ts['city']==c) & (ts['outcome']==o)"
                    col='agg_per_1000'
                aux = newcols
                f = ts[ eval(cond) & (ts['type_period']=='week') ][ ['period', 'agg_per_1000', 'residential'] ]
                y = f[col].fillna(0).values
                i=0
                for k in f['period'].values:
                    aux[k] = y[i]
                    i+=1
                xs[o].append( [ float(el) for el in list(aux.values()) ] )
            print(ind, '/', len(cities))
            ind+=1
    
        with open('x_clustering.json', 'w') as g:
            json.dump(xs, g)
    else:
        with open('x_clustering.json', 'r') as g:
            xs = json.load(g)
            
    return xs

# ---------- Clustering by graph ---------------
def cluster_by_network(cities, xs, cutoff):
    g = open("results_clustering_network.tsv","w")
    g.write(f"identifier\tid_cluster\tnumber_elements\tcities\n")
    g.close()
    for ide in xs.keys():
        m = distance.cdist( xs[ide], xs[ide], 'euclidean' )
        g = nx.Graph()
        
        i=0
        for x in m:
            c1 = cities[i]
            j=0
            for y in x:
                c2 = cities[j]
                if(i<j):
                    if( y < cutoff ):
                        g.add_edge(c1, c2)
                j+=1
            i+=1
        
        coms = nx.connected_components(g)
        i=1
        for c in coms:
            els = ','.join(c)
            with open("results_clustering_network.tsv","a") as gf:
                gf.write( f"{ide}\tc-{i}\t{len(c)}\t{els}\n")
            i+=1

def run_simulation_network():
    ts = pd.read_csv('../filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
    cities = list(ts['city'].unique())
    
    xs = make_input()
    
    cluster_by_network(cities, xs, 10)    


# ---------- Clustering by ml kmeans ---------------
def generate_groups(X, k):
    m = KMeans(n_clusters=k, random_state=0).fit(X)
    y = m.labels_
    return y

def test_elbow_method(xs, cutoff):
    res={}
    
    g = open("results_clustering_elbow.tsv","w")
    g.write(f"identifier\tk\tdistortion\tinertia\tratio_inertia\n")
    g.close()
    for ide in xs.keys():
        X = np.array(xs[ide])
        
        os.system(f"rm -rf clustering_{ide.lower()}")
        os.system(f"mkdir clustering_{ide.lower()}")
        
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = range(2, 40)
        
        ant= 0
        best_k = 0
        ratio = 1
        antine=0
        for k in K:
            # Building and fitting the model
            m = KMeans(n_clusters=k, random_state=0).fit(X)
            m.fit(X)
            
            dist = sum(np.min(cdist(X, m.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0] # calculates the minimum distance mean among the data records coordinates and the center points of the found clusters, it returns the mean considering all city records
            distortions.append(dist)
            mapping1[k] = dist
            
            ine = m.inertia_
            inertias.append(ine)
            mapping2[k] = ine
            
            if(antine==0):
                best_k = k
            else:
                ratio = (antine-ine)/antine
                #dif = ratio-ant
                if( ratio <= cutoff ): # Once it achieves the a minimum difference in the inertia reduction percentage among the steps according to cutoff it stops
                    best_k = k
                    break
                    
            print(ide, k, antine, ine, ant, ratio)
            ant=ratio
            antine=ine
            
            with open("results_clustering_elbow.tsv","a") as g:
                g.write(f"{ide}\t{k}\t{dist}\t{ine}\t{ratio}\n")
                
        K = range(2, best_k+1)
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.savefig(f"clustering_{ide.lower()}/distortions.png")
        plt.clf()
        
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method using Inertia')
        plt.savefig(f"clustering_{ide.lower()}/inertias.png")
        plt.clf()
        
        y = generate_groups(X, best_k)
        
        pca = PCA(n_components=2)
        xt = pca.fit_transform(X)
        plt.scatter(xt[:, 0], xt[:, 1], c=y)
        plt.title('K-means clustering (k={})'.format(k))
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.savefig(f"clustering_{ide.lower()}/best_k-{best_k}_points_distribution.png")
        plt.clf()
        
        res[ide] = { 'y': y, 'best_k': best_k }
        
    return res
    
def run_simulation():
    g = open("results_clustering_scenarios_outcome_from_mobility.tsv", "w")
    g.write("type_scenario\tvariable_column_x\tcity\tpredicted_cluster\n")
    g.close()
    
    # Scenario 1 - outcomes with cases
    # Scenario 2 - outcomes with mobility
    
    xs = make_input()
    
    res_elbow = test_elbow_method(xs, 0.01)
    for o in xs.keys():
        X = xs[o]
        col='residential'
        typ = o
        if( o!='mobility' ):
            col=o
            typ='srag'
        #y = generate_groups(X)
        r = res_elbow[o]
        print(o, r['best_k'])
        y = r['y']
        i=0
        for j in y:
            with open("results_clustering_scenarios_outcome_from_mobility.tsv", "a") as g:
                g.write(f"{typ}\t{col}\t{cities[i]}\t{j}\n")
            i+=1

# ---------- Pos processing analysis ---------------
def get_closest_cluster_by_neighbor(sett, set_group):
    ma = -1
    best_k = 0
    els=[]
    for k in set_group.keys():
        inter = sett.intersection(set_group[k])
        if(len(inter)>ma):
            ma = len(inter)
            best_k = k
            els = list(inter)
    return best_k, ma, els
    
def get_metrics_intra_cluster(dat, els):
    outcomes = ['Cases','Deaths', 'mobility']
    means = {}
    for e in els:
        for o in outcomes:
            if(not o in means):
                means[o]=[]
            means[o].append( dat[e][o] )
    
    strm=[]
    for o in outcomes:
        temp = means[o]
        means[o] = sum(temp)/len(temp)
        strm.append( str( means[o] ) )
    strm = '\t'.join(strm)
    return means, strm
        
        
def check_intersection_dimensions():
    ts = pd.read_csv('../filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
    cities = list(ts['city'].unique())
    with open('x_clustering.json', 'r') as g:
        xs = json.load(g)
    dat={}
    for c in cities:
        dat[c]={}
    for k in xs.keys():
        i=0
        for c in cities:
            dat[c][k] = sum(xs[k][i])/len(xs[k][i])
            i+=1        
           
    outs=['Cases','Deaths']
    df = pd.read_csv("results_clustering_scenarios_outcome_from_mobility.tsv", sep='\t')
    dtct={}
    dtcol={}
    for i in df.index:
        col = df.loc[i, 'variable_column_x']
        cl = df.loc[i, 'predicted_cluster']
        cls = f"cl-{cl}"
        ct = df.loc[i, 'city']
        
        if(not col in dtcol):
            dtcol[col]={}
        if(not cls in dtcol[col]):
            dtcol[col][cls]=set()
        dtcol[col][cls].add(ct)
        
        if(not ct in dtct):
            dtct[ct]={}
        dtct[ct][col]=cl
    
    g = open("results_analysis_clustering_scenarios.tsv", "w")
    g.write("outcome\tmaximum_intersection\tpercentage_intersection\toutcome_cluster\toutcome_cluster_elements\tomeans_cases\tomeans_deaths\tomeans_mobility\tresidential_cluster\tresidential_cluster_elements\trmeans_cases\trmeans_deaths\trmeans_mobility\n")
    g.close()
    
    cltsr = dtcol['residential']
    matches={}
    for o in outs:
        matches[o]={}
        clts = dtcol[o]
        for c in cltsr.keys():
            cl = cltsr[c]
            rmeans, srm = get_metrics_intra_cluster(dat, cl)
            
            scl=','.join( list(cl) )
            best_k, ma, els = get_closest_cluster_by_neighbor(cl, clts)
            matches[o][c] = { 'best_cl': best_k, 'number_elements': ma, 'elements': els }
            gmeans, sgm = get_metrics_intra_cluster(dat, els)
            
            sels = ','.join( list(els) )
            with open("results_analysis_clustering_scenarios.tsv", "a") as g:
                g.write(f"{o}\t{ma}\t{ma/len(cl)}\t{best_k}\t{sels}\t{sgm}\t{c}\t{scl}\t{srm}\n")
            

import sys
option = sys.argv[1]
if(option=='1'):
    run_simulation()
if(option=='2'):
    run_simulation_network()
if(option=='3'):
    check_intersection_dimensions()  
