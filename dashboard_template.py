#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
# from sklearn.preprocessing import LabelEncoder, StandardScaler


#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Prince Jeffrey Villamil\n2. Name\n3. Name\n4. Name\n5. Name")

#######################
# Data

# Load data
dataset = pd.read_csv("data/ds_salaries.csv")
df = dataset.copy()
df.dropna(axis=1, how='any')
dfnew = df.drop(columns=['Unnamed: 0'])
dfnewCopy = dfnew.copy()

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.markdown(""" 

    In this group project, we want to predict the possible average data scientist salaries along with experience level for said salaries, and if possible we would like to see if it can give insight into the future job market. We believe this is the best data set to use for this because the job market for tech has been in an influx thus having a data set that focuses on years where this has occurred will give better results in our productions.

    #### Pages

    1. `Dataset` - Overview of a data science job salary database, including work year , experience level, employment type, company locations, and additional influencing.
    2. `EDA` - Exploratory analysis on salary distributions and demographic patterns within the data science industry, with visualizations highlighting correlations between experience, location, and job type.
    3. `Data Cleaning / Pre-processing` - Cleaning and transforming the salary data to ensure accuracy in modeling, handling missing or inconsistent values, and selecting relevant columns for analysis.
    4. `Machine Learning` - Implementing Regression Analysis to predict salary based on factors such as job title, experience, and location, as well as other relevant variables.
    5. `Prediction` - Prediction feature to estimate potential data science job salaries, providing insights based on past data and critical salary-influencing factors.
    6. `Conclusion` - Summarized findings on data science job salary trends, key factors influencing salaries, and model effectiveness in forecasting salary predictions.
                """)

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    # st.write("Data Science Job Salaries Statistics Dataset Overview")
    
    st.markdown("""
        ### Data Science Job Salaries Statistics Dataset Overview
                
        **Link to dataset**: [Data Science Job Salaries Data Set on Kaggle](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
                
        The dataset is about Data Science Job Salaries, it highlights different areas that will contribute to a Data Scientist's salaries, such as number of work years, experience level, and types of employment. This data set seems like it is set from 2020-current date, which should give us the current landscape of data science jobs, which is really valuable in being able to predict outcomes such as future salaries and the future job market. For our group project, we want to predict the possible average data scientist salaries along with experience level for said salaries, and if possible we would like to see if it can give insight into the future job market. We believe this is the best data set to use for this because the job market for tech has been in an influx thus having a data set that focuses on years where this has occurred will give better results in our productions. This data will be very interesting to see because as we aspire to join the job market for data science/tech jobs, we can have a better grasp for what we will be confrutned with once we graduate. Lastly, models to use on the dataset, since we want to predict using historical data, we want to use models focused on years of experience x year x year salary. Based on what we have searched, time series models are what we want so for things like the exploratory data analysis models, we also want to experiment with using a regression analysis model to see which factors contribute the most when predicting salaries based on work_year, experience_level, job_title, and etc.

        ### Proposed Models
            Given the multivariate nature of this prediction, we will employ a linear regression model which would be used to test which features would impact data science job salary the most in our machine learning predictions.

        ### Dataset Preview
                """)
    st.dataframe(dfnew, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Descriptive Statistics
    """)
    st.dataframe(dfnew.describe(), use_container_width=True)


    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.dataframe(dfnew, use_container_width=True, hide_index=True)
    st.dataframe(dfnew.describe(), use_container_width=True)

    remote_ratio_counts = dfnewCopy['remote_ratio'].value_counts()
    custom_labels = {
        0: "Less than 20%",
        50: "Partially Remote (50%)",
        100: "Fully Remote (More than 80%)"
    }
    labels = [custom_labels[val] for val in remote_ratio_counts.index]
    pie_data = pd.DataFrame({
        'Labels': labels,
        'Counts': remote_ratio_counts.values
    })
    fig = px.pie(pie_data, names='Labels', values='Counts', title='Distribution of Remote Work Ratio')
    st.plotly_chart(fig)
    avg_salary_by_size = dfnewCopy.groupby('company_size')['salary_in_usd'].mean()
    avg_salary_df = pd.DataFrame({
        'Company Size': avg_salary_by_size.index,
        'Average Salary in USD': avg_salary_by_size.values
    })
    st.title("Average Salary by Company Size")
    st.bar_chart(avg_salary_df.set_index('Company Size'))




    # col = st.columns((1.5, 4.5, 2), gap='medium')

    # # Your content for the EDA page goes here

    # with col[0]:
    #     st.markdown('#### Graphs Column 1')


    # with col[1]:
    #     st.markdown('#### Graphs Column 2')
        
    # with col[2]:
    #     st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    st.dataframe(dataset.head(), use_container_width=True, hide_index=True)
    st.markdown("Currently, this dataset includes columns that are not relevant to this project. To address this, we have refine the dataset by selecting only the necessary columns and creating a new dataset containing just the required data, which can be seen in the next table.")
    #Deletes 'Unnamed: 0' Column from original df
    dfnew = df.drop(columns=['Unnamed: 0'])
    # Columns with String values
    cat_col = [col for col in dfnew.columns if dfnew[col].dtype == 'object']
    # print('Categorical columns :',cat_col)
    # Columns with Int values
    num_col = [col for col in dfnew.columns if dfnew[col].dtype != 'object']
    # print('Numerical columns :',num_col)
    st.dataframe(dfnew.head(), use_container_width=True, hide_index=True)
    st.code("fnew = df.drop(columns=['Unnamed: 0'])")
    st.markdown("This line of code will drop/remove unwanted columns.")
    st.code("cat_col = [col for col in dfnew.columns if dfnew[col].dtype == 'object']")
    st.code("[col for col in dfnew.columns if dfnew[col].dtype != 'object']")
    st.markdown("This will process the dataset and organize the data into two categories: one for columns with string values and the other for columns with integer values.")

    col = st.columns((1, 1), gap='medium') 
    with col[0]:
        st.markdown("Columns with String values")
        st.dataframe(cat_col, use_container_width=True, hide_index=True)

    with col[1]:
        st.markdown("Columns with Int values")
        st.dataframe(cat_col, use_container_width=True, hide_index=True)



    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here