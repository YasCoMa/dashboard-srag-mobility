# Statistical and machine learning methods to handle time series analysis

## Summary
This folder contains three algorithms to perform the following analysis: 
- [outcome-prediction] Test capability of the residental percentage change combined with some other place category or alone to predict the overall cases using regression approaches.
- [wave-detection] Detect the length and properties of waves observed in a numerical series and the trends step by step.
- [city-clustering] Perform two strategies of clustering to group cities by their behavior pattern of cases, deaths or residential percentage change along the week time period. These strategies rely on (i) Connected components detection in a graph formed by cities as nodes and the euclidean distance among their features as edges. And (ii) traditional ML unsupervised algorithm (KMeans), optimized with elbow method to find the best number of groups. 

## Requirements:
* Python packages needed:
	- pip3 install sklearn
	- pip3 install plotly
	- pip3 install matplotlib
	- pip3 install pandas
	- pip3 install scipy
	- pip3 install numpy
	- pip3 install networkx
	- pip3 install statsmodels

## Usage Instructions
* [outcome-prediction] :
    - **Command**: ````python3 predict_outcome_y_from_mobility.py````
	- **Output**: results_regression_outcome_from_mobility.tsv
	
* [wave-detection] :
    - **Command**: ````python3 detect_cycles_ts.py````
	- **Outputs**: waves_cities.tsv and not_detected_wave.tsv
	
* [city-clustering] :
    - Using kmeans and elbow strategies:
        - **Command**: ````python3 clustering_by_outcome_mobi.py 1````
	    - **Output files**: results_clustering_elbow.tsv and results_clustering_kmeans.tsv
	    - **Output folders** (for the grouped points visualization plots): clustering_cases, clustering_deaths and clustering_mobility
	    
    - Using graph strategy:
        - **Command**: ````python3 clustering_by_outcome_mobi.py 2````
	    - **Output file**: results_clustering_network.tsv
    
    - Post processing analysis for kmeans:
        - **Command**: ````python3 clustering_by_outcome_mobi.py 3````
	    - **Output files**: results_summary_clusters_kmeans.tsv and results_analysis_clustering_kmeans.tsv
    
    - Post processing analysis for graph:
        - **Command**: ````python3 clustering_by_outcome_mobi.py 3````
	    - **Output files**: results_summary_clusters_network.tsv and results_analysis_clustering_network.tsv
	
