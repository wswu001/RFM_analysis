##### Import libraries #####
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

##### Read data  #####
df = pd.read_csv('cdp_data.csv').iloc[:, :12]
df['purchasedate'] = pd.to_datetime(df['purchasedate'], format='%Y/%m/%d')

##### Create RFM table ("result") #####
# Group by 'cdpid' and perform aggregations
result = df.groupby('cdpid').agg(
    frequency=pd.NamedAgg(column='purchasedate', aggfunc='size'),
    recent_purchase=pd.NamedAgg(column='purchasedate', aggfunc='max'),
    monetary=pd.NamedAgg(column='total_price', aggfunc='sum')
).reset_index()

# Calculate recency as the difference between the maximum purchasedate and today's date
result['recency'] = (datetime.now()-result['recent_purchase']).dt.days

# Drop unnecessary column
result =result.drop(['recent_purchase'], axis =1)


##### Perform log transformation and save to "df_log" #####
df_log = result.copy()
df_log['frequency'] = np.log(df_log['frequency']+1)
df_log['monetary'] = np.log(df_log['monetary']+1)
df_log['recency'] = np.log(df_log['recency']+1)


##### Standardize df_log #####
X = df_log.iloc[:,1:]
scaler = StandardScaler()
scaler.fit(X)
RFM_scaled = scaler.transform(X)
RFM_scaled = pd.DataFrame(RFM_scaled, columns=X.columns)


##### Modeling: Kmeans & Elbow method #####
K = range(1,10)
wcss = []
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k, random_state = 2024).fit(RFM_scaled) 
    kmeanModel.fit(RFM_scaled)   
    wcss.append(kmeanModel.inertia_)
plt.plot(K, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  
plt.show()

##### Build Kmeans model #####
def create_kmeans(K, standardized_data):
    # Build Kmeans model with certain number of cluster
    Model = KMeans(n_clusters=K, random_state = 2024)

    # Get cluster label and concate on original RFM data
    cluster_labels = Model.fit_predict(standardized_data)
    result_new = result.copy()
    result_new['Cluster']= cluster_labels

    # Plot snake plot
    RFM_scaled_splot = standardized_data.copy()
    RFM_scaled_splot['Cluster'] = cluster_labels

    result_splot = RFM_scaled_splot.groupby('Cluster').agg(
        frequency=pd.NamedAgg(column='frequency', aggfunc='mean'),
        recency=pd.NamedAgg(column='recency', aggfunc='mean'),
        monetary=pd.NamedAgg(column='monetary', aggfunc='mean'))
    
    # Melt the DataFrame for snake plot
    result_splot_melted = pd.melt(result_splot.reset_index(), 
                                id_vars=['Cluster'], 
                                var_name='Metric', 
                                value_name='Mean Value')

    # Create a snake plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Metric', y='Mean Value', hue='Cluster', 
                 data=result_splot_melted, marker="o", ci=None)
    plt.xlabel('Metrics')
    plt.ylabel('Mean Values')
    plt.title('Snake Plot')
    plt.show()

    # Plot RFM table with mean value for each cluster 
    result_new_mean = result_new.iloc[:,1:].groupby('Cluster').agg(
        frequency=pd.NamedAgg(column='frequency', aggfunc='mean'),
        recency=pd.NamedAgg(column='recency', aggfunc='mean'),
        monetary=pd.NamedAgg(column='monetary', aggfunc='mean'))
    
    return result_new_mean