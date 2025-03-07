# SAFFRON SKIES AIRWAY PASSENGER ANALYSIS
![](https://github.com/user-attachments/assets/e2ad18c1-63f1-4436-ba0d-e06af04babdf)

# Overview
In the competitive airline industry, passenger satisfaction is crucial for success. This report for Saffron Skies Airways analyzes extensive passenger satisfaction data to identify key factors affecting customer experiences. It aims to provide actionable insights for service improvement using machine learning and advanced data visualization techniques. 

# Objectives
- Identify key factors influencing passenger satisfaction.
- Apply an ETL (Extract, Transform, Load) process to ensure data quality.
- Perform Exploratory Data Analysis (EDA) to uncover trends and correlations.
- Use Random Forest Classifier to predict satisfaction levels.
- Evaluate model performance using confusion matrix, accuracy, precision, recall, and feature importance.

# Tools & Technologies Used
- Python for data processing and analysis.
- Pandas, NumPy for data manipulation.
- Matplotlib, Seaborn for data visualization.
- Scikit-learn for machine learning models.
- Google Colab for code execution.

# Data Processing Steps

## **1. Extract, Transform, Load (ETL)**

The ETL process is crucial for ensuring data quality and reliability. We began by extracting the dataset containing over 103,000 records on passenger satisfaction from a CSV file.
During the transformation phase, I handled missing values using the median value to fill them up, ensuring data continuity without bias.
Categorical variables such as 'Gender' and 'Type of Travel' were encoded into numerical values using LabelEncoder to prepare them for machine learning algorithms. 
Outliers were identified and analyzed using IQR and box plots, meaningful outliers were retained, while erroneous ones were removed to maintain data integrity.

## **2. Exploratory Data Analysis (EDA)**
Exploratory Data Analysis (EDA) is essential in identifying anomalies, such as missing values and outliers, and ensures that the data is clean and suitable for analysis.
Visualizations, such as histograms, box plots, and heatmaps, were used to make complex data more accessible and to highlight relationships between variables that are not immediately apparent from raw data 

 **- a. Top 10 Destinations**

 India is the most frequented destination which is very overwhelming

![](https://github.com/user-attachments/assets/b0f7fa90-cd2c-4670-a1fd-0ae425ce177b)

**- b. Travel Type**

30000 customers used the airline for personal travels , like vacation, shopping ,etc while more than double the number used it for business purposes

![](https://github.com/user-attachments/assets/83488d5a-7e5c-4c1e-bf0f-463711ae1a41)

**- c. Airline Usage by Gender**

Almost equal distribution of gender that use the airline

![](https://github.com/user-attachments/assets/7a79ea3a-ea82-4d31-87ba-86a82aab5cfa)

**- d. Passenger Satisfaction**

80% of customers are satisfied across our services, then over 20% are not happy with the service they are getting

![](https://github.com/user-attachments/assets/989cb389-7bf7-49a5-a9cd-7d6aea4ed61e)

**- e. Age Distribution**

Age distribution is a normal distribution with the mean age of travellers being around 40 years

![](https://github.com/user-attachments/assets/40647dbf-3164-4583-b979-9ac0a1bcf309)

**- f. Passenger Class Distribution**

Business class had the higher patronage of 47.87%, followed closely by Eco, 45.0% and lastly Eco plus, 7.2%

![](https://github.com/user-attachments/assets/a3975d4a-765a-4262-8820-4268d23f5ea3)

**- g. Customer Satisfaction Across Continent**

Asia had the higest dis-satisfied customers of over 65%, but also the highest patronage compared to the others , followed by Europe
Africa seems to have the lowest dis-satisfaction rate.

![](https://github.com/user-attachments/assets/5fb521b1-5b49-4f89-8147-3d8a7af157fd)

**- h. Arrival-Departure vs Satisfaction-Convinient**

There is a strong postive correlation between the aircraft departure time and arrival time i.e the higher the delay in departure and higher the arrival time for arrival at destination, which is obviously valid
There is no clear definition if the relationship between the two variable affect the customer satisfaction
Departure and arrival time convienient increases the satisfaction of customers

![](https://github.com/user-attachments/assets/0a1ae2ae-6243-4761-a8dd-7f278a8faeec)

**- i. Key factors affecting customer experiences.**

The airline provided great services in Baggage handling, inflight service and enterntainment and on-board service, which leaded to more satisfied customers
Poor performance in inflight wifi service, Ease of online booking and gate location

![](https://github.com/user-attachments/assets/b1310076-2011-4230-972b-6e81d2a45eee)
![](https://github.com/user-attachments/assets/083ab7b6-5cfb-47db-866f-f652e8198317)

**- j. Correlation Matrix**

![](https://github.com/user-attachments/assets/ee8c3ab8-56f2-4b3e-ae67-0b0942d07b02)

Identifying a strong correlation between in-flight service and overall satisfaction suggests that regression models could be effective. 

## **3. Model Building**


