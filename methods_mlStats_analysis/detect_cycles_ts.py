import pandas as pd

def find_waves_cities():
    g=open("waves_cities.tsv","w")
    g.write("city\ttotal_waves\twave_order\tpeak\tindexes\tdates\tcase_numbers\n")
    g.close()
    
    fg=open("not_detected_wave.tsv","w")
    fg.write("city\tnumber_periods\tcases\n")
    
    window=3
    
    ts = pd.read_csv('../filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
    ts = ts[ (ts['outcome']=='Cases') & ( (ts['period'].str.contains('2020') ) | (ts['period'].str.contains('2021')) ) & (ts['type_period']=='week') ]
    if( len(ts) > window*2):
        cities = ts['city'].unique()
        k=1
        for c in cities:
            
            f = ts[ (ts['city']==c) & (ts['outcome']=='Cases') ]
            s = f[ ( (f['period'].str.contains('2020') ) | (f['period'].str.contains('2021')) ) & (f['type_period']=='week') ][['period','agg']]
            s['date'] = pd.to_datetime( s['period'] + '-1', format='%Y-w%W-%w')
            se = s['agg'].values
            w = detect_waves(se, window)
            i=1
            n=len(w)
            for el in w:
                peak = el[0]
                inds = el[1]
                dates = s['period'].values[ inds ]
                dates = ','.join(dates)
                
                sinds = [ str(j) for j in inds ]
                sinds = ','.join(sinds)
                
                ncases = [ str(se[j]) for j in inds ]
                ncases = ','.join(ncases)
                
                with open("waves_cities.tsv","a") as g:
                    g.write(f"{c}\t{n}\t{i}\t{peak}\t{sinds}\t{dates}\t{ncases}\n")
                i+=1
            k+=1
            
            if(n==0):
                print(c, len(se) )
                fg.write(f"{c}\t{len(s)}\t{sum(se)}\n")
    fg.close()

def detect_waves(cases, window):
    vs=cases
    
    ci=0
    general=[]
    for v in vs:
        if( ci < len(vs)-1 ):
            curr = vs[ci]
            next = vs[ci+1]
            
            trend='up'
            if( curr > next):
                trend='down'
                
            caux = ci
            general.append( [curr, next, caux, trend] )
            
        ci+=1
    
    cup=0
    i=0
    waves=[]
    inds=[]
    while i < len(general):
        it=general[i]
        if( it[3]=='up' ):
            cup+=1
            inds.append(i)
        else:
            flagdown=True
            k=i+1
            if( k<len(general) ):
                if(general[k][3]=='up'):
                    flagdown=False
                    cup+=1
                    inds.append(k)
                    
            if(flagdown and cup >= window):
                peak = it[0]
                j=i
                if( i < len(general)-window ):
                    j=i+1
                    cdown=1
                    while j<len(general):
                        if( general[j][3]=='down' ):
                            cdown+=1
                            inds.append(j)
                        else:
                            flagup=True
                            k=j+1
                            if( k<len(general) ):
                                if(general[k][3]=='down'):
                                    flagup=False
                                    cdown+=1
                                    inds.append(j)
                            if(flagup and cdown >= window+1):
                                waves.append( [peak, inds] )
                                inds=[]
                                cup=1
                                cdown=0
                                i=j
                                break
                        j+=1
                    
                                
                inds=[]
                cup=1
                cdown=0
                if(j>i):
                    i=j
            else:
                if(flagdown and k>i):
                    i=k   
        i+=1
    return waves

print("---- waves per city")
find_waves_cities()
