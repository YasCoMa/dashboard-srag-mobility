import streamlit as st
import pandas as pd
import plotly.express as px

from analysis_eda_srag import DashboardEdaSrag
from analysis_eda_mobility import DashboardEdaMobi
from analysis_eda_citycluster import DashboardClustering

def run_UI():
    st.set_page_config(
        page_title="Analysis of Brazilian COVID-19 data + Google mobility change reports",
        page_icon="🦠",
        menu_items={
            'Report a bug': "https://github.com/YasCoMa/dashboard-streamlit-sragmobi/issues/new/choose",
            'About': """            
         If you're seeing this, we would love your contribution! If you find bugs, please reach out or create an issue on our 
         [GitHub](https://github.com/YasCoMa/dashboard-streamlit-sragmobi) repository. If you find that this interface doesn't do what you need it to, you can create an feature request 
         at our repository or better yet, contribute a pull request of your own. 
    
         Dashboard Creator: Yasmmin Martins
        """
        }
    )
    st.sidebar.title('Srag-Mobility Data Analysis')
    st.sidebar.write("""
            ## About
            
            - Analysis of some dimensions of the SRAG data to explore some of the insights that can be extracted along with the time series provided by the mobility data.
            - Data quality measure to report the missing/wrong data in both datasets
            - Cities clustering experiment to test the ability to classify sucessful or not cases where the protection measures were well implemented to control the disease.
        """)
        
    clean_srag = pd.read_csv('filtered_data/srag_data.tsv.gz', sep='\t', compression='gzip')
    clean_mobi = pd.read_csv('filtered_data/report_mobility.tsv.gz', sep='\t', compression='gzip')
    clean_ts = pd.read_csv('filtered_data/time_series_mobility_cases.tsv.gz', sep='\t', compression='gzip')
    #db = clean
    #clean.dropna()
    
    main = st.container()
    with main:
        st.markdown("## [Data sources information] - This tool performs a series of exploration analysis in two datasets:")
        col1, col2 = st.columns(2)
        with col1:
            st.header("srag")
            st.markdown("[Severe Accute Respiratory Syndrome - SUS](https://opendatasus.saude.gov.br/dataset/srag-2021-a-2023) - This an open database provided by the federal government collected along the care units of the Health Unique System (SUS - Brazilian FREE (:-D) healthcare plane). The dataset contains individual cases reported with a lot of column details describing the risk factors of the patients, gender, birth date, race, pregnancy status, symptoms and other data")
        with col2:
            st.header("mobility")
            st.markdown("[Google Community mobility Reports](https://www.google.com/covid19/mobility/) - Google provides unified reports compiled by the pattern change observed in the usage of the mobile devices in the place types of google maps, categorized mainly by Retail and recreation, Grocery and pharmacy, parks, transit stations, workplaces and residential.")
        
        tab1, tab2, tab3 = st.tabs(["Brazilian COVID-19 notifications exploration", "Mobility changes data exploration", "City clustering"])
      
        with tab1:
            st.header("Exploring SRAG dataset")
            obj = DashboardEdaSrag()
            obj.eda_UI(clean_srag)
        with tab2:
            st.header("Exploring Mobility change dataset")
            obj = DashboardEdaMobi()
            obj.eda_UI(clean_mobi, clean_ts)
        
        with tab3:
            #st.header("Cities network")
            obj = DashboardClustering()
            obj.eda_UI(clean_ts)
        
        
if __name__ == '__main__':
    run_UI()    
