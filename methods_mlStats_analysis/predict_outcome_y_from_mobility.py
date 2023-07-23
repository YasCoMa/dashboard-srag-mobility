import math
import pandas as pd
import statsmodels.api as sm
import numpy as np

def test_glm_fit(x, y):
    m = sm.GLM( y, x, family=sm.families.Gamma())
    r = m.fit()
    ys = np.array( y.values )
    diff = ys-r.fittedvalues
    mse = np.square( diff ).mean()
    rmse = math.sqrt( mse )
    md = diff.mean()
    return rmse, md
    
def test_ols_fit(x, y):
    m = sm.OLS( y, sm.add_constant(x, prepend=False))
    r = m.fit()
    params = r.params
    ys = np.array( y.values )
    diff = ys-r.fittedvalues
    mse = np.square( diff ).mean()
    rmse = math.sqrt( mse )
    md = diff.mean()
    return rmse, md, params
        
def run_simulation():
    ts = pd.read_csv('../filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
    cities = ts['city'].unique()
    outcomes = ts['outcome'].unique()
    
    g = open("results_regression_outcome_from_mobility.tsv", "w")
    g.write("city\toutcome\tx_cols\ty_col\trmse_residual_glm\trmse_residual_ols\tmd_residual_glm\tmd_residual_ols\tguest_x_col\tols_coefficient_residential\tols_coefficient_guest_x_col\n")
    g.close()
    
    cols_y = [ 'agg_per_1000', 'agg_per_1000_log10']
    cols_x = ['', 'retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','workplaces']
    for c in cities:
        for o in outcomes:
            print(c, o)
            f = ts[ (ts['city']==c) & (ts['outcome']==o) & ( (ts['period'].str.contains('2020') ) | (ts['period'].str.contains('2021')) ) & (ts['type_period']=='week') ][['period','agg', 'agg_per_1000', 'agg_per_1000_log10', 'retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','workplaces','residential']]
            for cx in cols_x:
                x_cols = ['residential']
                if(cx!=''):
                    x_cols.append(cx)
                else:
                    cx='-'
                    
                x = f[ x_cols ]
                x_cols = ','.join(x_cols)
                x = x.fillna(0)
                
                for cy in cols_y:
                    y = f[cy].fillna(0)
                    
                    try:
                        rmseg, mdg = test_glm_fit(x, y)
                        rmse, md, params = test_ols_fit(x, y)
                        guest_cf = '-'
                        if(cx!='-'):
                            guest_cf = params[cx]
                        with open("results_regression_outcome_from_mobility.tsv", "a") as g:
                            g.write(f"{c}\t{o}\t{x_cols}\t{cy}\t{rmseg}\t{rmse}\t{mdg}\t{md}\t{cx}\t{params['residential']}\t{guest_cf}\n")
                    except:
                        pass
                        
run_simulation()                        
