import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

st.title('4baseCare CDSS support')

ddf = pd.read_csv('combi_data_ca1.csv', index_col=0)


df = ddf.copy()
scaler = MinMaxScaler()
df[['age', 'MSI']] = scaler.fit_transform(df[['age', 'MSI']])
df['gender'] = df['gender'].map({'MALE': 0, 'FEMALE': 1}) #encoding 1
genes_df = df[ddf.drop(['age', 'gender', 'cancerSite', 'morphologyIdcCode'], axis=1).columns.to_list()]
additional_features_df = df[['age', 'gender']] ## Demographics
c_features_df = df[['cancerSite', 'morphologyIdcCode']] ## Cancer 
cancer_features_df = pd.get_dummies(c_features_df, columns=['cancerSite']).drop('morphologyIdcCode', axis=1) # encoding 2
morph_df = pd.get_dummies(c_features_df, columns=['morphologyIdcCode']).drop('cancerSite', axis=1) # encoding 3


# weightage
weight_genes = st.slider("Select Weightage for Genomic Data",10, 90, 80)
weight_additional = st.slider("Select Weightage for Clinical Data",10, 90, 4)
weight_cancer = st.slider("Select Weightage for Cancer Data",10, 90, 16)

w_can = weight_cancer / 2
w_morph = weight_cancer / 2

weighted_genes_df = genes_df * (weight_genes / 100)
weighted_additional_features_df = additional_features_df * (weight_additional / 100)
wtd_cancer = cancer_features_df * (w_can / 100)
wtd_morph = morph_df * (w_morph / 100 )
combined_df = pd.concat([weighted_genes_df, weighted_additional_features_df, wtd_cancer, wtd_morph], axis=1)

# Compute cosine similarity for the combined data
similarity_matrix = cosine_similarity(combined_df)

# Function to get the most similar patients
def get_similar_patients(patient_index, n):
    similarity_scores = similarity_matrix[patient_index]
    similar_indices = similarity_scores.argsort()[::-1][1:n+1]  # Exclude the patient itself
    return similar_indices

# Function to get common genes
def get_common_genes(patient_index, similar_patient_index):
    patient_genes = genes_df.iloc[patient_index].astype(bool)
    similar_patient_genes = genes_df.iloc[similar_patient_index].astype(bool)
    common_genes = patient_genes & similar_patient_genes
    return genes_df.columns[common_genes]

# Function to convert similar patients to JSON
def similar_patients_to_json(similar_patients):
    similar_patient_ids = combined_df.index[similar_patients].tolist()
    return json.dumps(similar_patient_ids)

# User input for patient selection and number of similar patients
st.sidebar.title('Patient Similarity Finder')
patient_id = st.sidebar.selectbox('Select Patient ID', combined_df.index)
num_similar_patients = st.sidebar.number_input('Number of similar patients to display', min_value=1, max_value=20, value=5)

# Function to create a DataFrame for similar patients
def create_similar_patients_df(similar_patients):
    data = []
    for idx in similar_patients:
        patient_info = ddf.iloc[idx]
        data.append({
            'Patient ID': combined_df.index[idx],
            'Cancer Site': patient_info['cancerSite'],
            'Morphology': patient_info['morphologyIdcCode'],
            'Age': patient_info['age'],
            'Gender': 'MALE' if patient_info['gender'] == 'MALE' else 'FEMALE',
            'MSI': patient_info['MSI']
        })
    return pd.DataFrame(data)



if st.sidebar.button('Find Similar Patients'):
    patient_index = combined_df.index.get_loc(patient_id)
    similar_patients = get_similar_patients(patient_index, num_similar_patients)
    
    # Display the selected patient's information
    st.subheader(f'Information for Patient {patient_id}')
    patient_info = ddf.loc[patient_id]
    patient_genes = genes_df.loc[patient_id].astype(bool)
    patient_common_genes = genes_df.columns[patient_genes]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Age:** {patient_info['age']}")
        st.markdown(f"**Gender:** {'MALE' if patient_info['gender'] == 'MALE' else 'FEMALE'}")
        st.markdown(f"**MSI:** {patient_info['MSI']}")
        st.markdown(f"**Cancer Site:** {patient_info['cancerSite']}")
        st.markdown(f"**Morphology** {patient_info['morphologyIdcCode']}")
    with col2:
        st.markdown(f"**Mutated Genes:** {', '.join(patient_common_genes)}")
    
    st.markdown("---")
    
    st.subheader(f'The {num_similar_patients} most similar patients to Patient {patient_id} are:')
    
    for idx in similar_patients:
        similar_patient_info = ddf.iloc[idx]
        common_genes = get_common_genes(patient_index, idx)
        age = similar_patient_info['age']
        gender = 'MALE' if similar_patient_info['gender'] == 'MALE' else 'FEMALE'
        MSI = similar_patient_info['MSI']
        cancer_site = similar_patient_info['cancerSite']
        
        with st.container():
            st.markdown(f"### Patient ID: {combined_df.index[idx]}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Age:** {age}")
                st.markdown(f"**Gender:** {gender}")
                st.markdown(f"**MSI:** {MSI}")
                st.markdown(f"**Cancer Site:** {cancer_site}")
            with col2:
                st.markdown(f"**Common Genes:** {', '.join(common_genes)}")
            st.markdown("---")
            
            
    # Create a DataFrame for similar patients
    similar_patients_df = create_similar_patients_df(similar_patients)
    st.subheader('Similar Patients DataFrame')
    st.dataframe(similar_patients_df)
         
    # Create JSON for similar patients and add download button
    similar_patients_json = similar_patients_to_json(similar_patients)
    st.download_button(
        label="Download Similar Patients as JSON",
        data=similar_patients_json,
        file_name=f'similar_patients_{patient_id}.json',
        mime='application/json'
    )
    
