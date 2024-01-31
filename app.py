# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

################ Modeling ################
# Read data
df = pd.read_csv('cdp_data.csv').iloc[:, :12]
df['purchasedate'] = pd.to_datetime(df['purchasedate'], format='%Y/%m/%d')

# Create RFM table ("result")
rfm_origin = df.groupby('cdpid').agg(
    frequency=pd.NamedAgg(column='purchasedate', aggfunc='size'),
    recent_purchase=pd.NamedAgg(column='purchasedate', aggfunc='max'),
    monetary=pd.NamedAgg(column='total_price', aggfunc='sum')
).reset_index()

rfm_origin['recency'] = (datetime.now()-rfm_origin['recent_purchase']).dt.days
rfm_origin = rfm_origin.drop(['recent_purchase'], axis=1)

# Perform log transformation and save to "df_log"
df_log = rfm_origin.copy()
df_log['frequency'] = np.log(df_log['frequency']+1)
df_log['monetary'] = np.log(df_log['monetary']+1)
df_log['recency'] = np.log(df_log['recency']+1)

# Standardize df_log
X = df_log.iloc[:, 1:]
scaler = StandardScaler()
scaler.fit(X)
rfm_scaled = scaler.transform(X)
rfm_scaled = pd.DataFrame(rfm_scaled, columns=X.columns)

# Define functions for Kmeans model and Snake Plot
def create_kmeans(K, standardized_data):
    model = KMeans(n_clusters=K, random_state=2024)
    cluster_labels = model.fit_predict(standardized_data)
    rfm_new = rfm_origin.copy()
    rfm_new['Cluster'] = cluster_labels

    # Plot snake plot
    rfm_scaled_splot = standardized_data.copy()
    rfm_scaled_splot['Cluster'] = cluster_labels

    splot = rfm_scaled_splot.groupby('Cluster').agg(
        frequency=pd.NamedAgg(column='frequency', aggfunc='mean'),
        recency=pd.NamedAgg(column='recency', aggfunc='mean'),
        monetary=pd.NamedAgg(column='monetary', aggfunc='mean'))

    # Melt the DataFrame for snake plot
    splot_melted = pd.melt(splot.reset_index(),
                                  id_vars=['Cluster'],
                                  var_name='Metric',
                                  value_name='Mean Value')

    # Create a snake plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Metric', y='Mean Value', hue='Cluster',
                 data=splot_melted, marker="o", ci=None)
    plt.xlabel('Metrics')
    plt.ylabel('Mean Values')
    plt.title('Snake Plot')
    st.pyplot()

    # Plot RFM table with mean value for each cluster
    rfm_new_mean = rfm_new.iloc[:, 1:].groupby('Cluster').agg(
        frequency=pd.NamedAgg(column='frequency', aggfunc='mean'),
        recency=pd.NamedAgg(column='recency', aggfunc='mean'),
        monetary=pd.NamedAgg(column='monetary', aggfunc='mean'),
        count=pd.NamedAgg(column='frequency', aggfunc='size'))
    return rfm_new, rfm_new_mean
    #

################ Streamlit App ################
# Introduction Markdown
st.markdown(f'<p style="background-color:#0066cc;color:#ffffff;font-size:18px;border-radius:2%;">{"AI Machine Learning：RFM Analysis using K-means Clustering"}</p>', unsafe_allow_html=True)
#st.title(':robot_face: RFM分群')
st.markdown(
    """
    此頁面使用AI中機器學習的方式來協助進行RFM分析，找出適當的臨界值來分出重要客戶。ITS旨在提供台灣區MarCom **CDP** 中重要隱含數據來做客戶分群的重要判斷。
    \n 
    :arrow_forward: **目標效益：**
    - 更精準的客戶分群：協助識別並理解不同客戶群體的行為特徵
    - 提升行銷ROI：深入理解客戶行為，協助MarCom更有效優化行銷活動，加大程度發揮行銷預算效益
    \n
    :arrow_forward: **數據來源：** cdp_loyalty_purchase 2017-2023年
    \n
    :arrow_forward: **使用方式：** 
    1. 在分群滑桿選擇想要的分群數(K)，最少1群，最多10群
    2. 根據分群數(K)參考
        - Snake Plot：線越沒有重疊，群體分得越明確
        - 3D Plot：顯示各群的散佈和區間內數值
        - 表格：顯示各群的中間值(平群值)以及數量
    """)
st.markdown("")


#st.set_option('deprecation.showPyplotGlobalUse', False)

# Elbow Method
wcss = []
for k in range(1, 11):
    KmeanModel = KMeans(n_clusters=k, random_state=2024).fit(rfm_scaled)
    KmeanModel.fit(rfm_scaled)
    wcss.append(KmeanModel.inertia_)

# Plot Elbow Method
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

st.subheader('1到10群的解釋力(Elbow Method)')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create slider for choosing K clusters
st.subheader('分群')
K = st.slider('請選擇要分成幾群：', 1, 10, 4)

#K-Means Model, Snake Plot, and RFM Table
st.subheader(f'K-Means Clustering with {K} clusters')
result_new, result_new_mean = create_kmeans(K, rfm_scaled)

st.subheader(f'RFM table with mean value in each cluster')
st.text("用下方圖表來幫助判斷客戶分群時的臨界點：")

# Plot 3D scatter plot using plotly
fig = px.scatter_3d(result_new, x='frequency', y='recency', z='monetary', color='Cluster',
                    labels={'frequency': 'Frequency', 'recency': 'Recency', 'monetary': 'Monetary'},
                    color_continuous_scale='Puor'
                    )

fig.update_layout(scene=dict(xaxis=dict(range=[0, 10]), #r
                             yaxis=dict(range=[0, 1500]), #f 
                             zaxis=dict(range=[-5, 150000])))#m 

st.plotly_chart(fig)

# Print RFM table
st.dataframe(result_new_mean,width=1200, height=600)

#st.markdown("")
