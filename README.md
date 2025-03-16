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
Analysis of customer-preferred destinations shows that Indida is the most frequent travel destination..

![](https://github.com/user-attachments/assets/b0f7fa90-cd2c-4670-a1fd-0ae425ce177b)

**- b. Travel Type**
The travel type distribution categorizes customers based on their travel purpose (business vs. leisure). This helps identify if satisfaction levels differ based on travel intent
30000 customers used the airline for personal travel, like vacation, shopping, etc, while more than double the number used it for business purposes

![](https://github.com/user-attachments/assets/83488d5a-7e5c-4c1e-bf0f-463711ae1a41)

**- c. Airline Usage by Gender**
The gender distribution analysis explores the proportion of male and female customers, assessing whether there are notable differences in satisfaction scores between genders

Almost equal distribution of gender that use the airline

![](https://github.com/user-attachments/assets/7a79ea3a-ea82-4d31-87ba-86a82aab5cfa)

**- d. Passenger Satisfaction**
Analysis of customer satisfaction levels shows the distribution of responses across different satisfaction categories. The visualization provides insights into how many customers are satisfied, neutral, or dissatisfied with the service
80% of customers are satisfied across our services, then over 20% are not happy with the service they are getting

![](https://github.com/user-attachments/assets/989cb389-7bf7-49a5-a9cd-7d6aea4ed61e)

**- e. Age Distribution**
The age distribution highlights the spread of customers across different age groups. This helps in understanding which age segments are most engaged with the service and how satisfaction levels vary across demographics.
Age distribution is a normal distribution with the mean age of travellers being around 40 years

![](https://github.com/user-attachments/assets/40647dbf-3164-4583-b979-9ac0a1bcf309)

**- f. Passenger Class Distribution**

Business class had the higher patronage of 47.87%, followed closely by Eco, 45.0%, and lastly, Eco plus, 7.2%

![](https://github.com/user-attachments/assets/a3975d4a-765a-4262-8820-4268d23f5ea3)

**- g. Customer Satisfaction Across Continent**

Asia had the higest dis-satisfied customers of over 65%, but also the highest patronage compared to the others, followed by Europe
Africa seems to have the lowest dissatisfaction rate.

![](https://github.com/user-attachments/assets/5fb521b1-5b49-4f89-8147-3d8a7af157fd)

**- h. Arrival-Departure vs Satisfaction-Convinient**
The departure and arrival factors illustrate how different locations influence customer satisfaction.
There is a strong positive correlation between the aircraft departure time and arrival time, i.e, the higher the delay in departure, the higher the arrival time for arrival at the destination, which is valid
There is no clear definition of whether the relationship between the two variables affects customer satisfaction
Departure and arrival time convienient increases the satisfaction of customers

![](https://github.com/user-attachments/assets/0a1ae2ae-6243-4761-a8dd-7f278a8faeec)

**- i. Key factors affecting customer experiences.**

The airline provided great services in Baggage handling, inflight service and enterntainment and on-board service, which leaded to more satisfied customers
Poor performance in inflight wifi service, Ease of online booking and gate location

![](https://github.com/user-attachments/assets/b1310076-2011-4230-972b-6e81d2a45eee)
![](https://github.com/user-attachments/assets/083ab7b6-5cfb-47db-866f-f652e8198317)

**- j. Correlation Matrix**
The correlation matrix shows statistical relationships between variables, highlighting which factors are most strongly associated with customer satisfaction.
Our analysis identified high correlations between class and type of travel (0.49), ease of online booking and inflight Wi-Fi (0.57), and seat comfort with inflight entertainment (0.59). Strong negative correlations include flight distance with destination (-0.65) and class with flight distance (-0.46).

![](https://github.com/user-attachments/assets/ee8c3ab8-56f2-4b3e-ae67-0b0942d07b02)

**- k. Factor Importance**

Factors influencing satisfaction were analyzed using feature importance techniques. The findings from logistic regression  and random forest models indicate key drivers of satisfaction, such as ease of booking, inflight wifi services, type of travel, and arrival/departure time

![](https://github.com/user-attachments/assets/17a87a7b-bf16-4a97-873f-b597170c1c3b)

![](https://github.com/user-attachments/assets/78b20204-0ef4-4388-ba8b-d2cfc4089f18)

## 3. Model Performance Comparison
The effectiveness of predictive models in classifying customer satisfaction was evaluated using:

**Random Forest Confusion Matrix** 

![](https://github.com/user-attachments/assets/27c8dd00-7620-453f-867f-5de5ef0c754f)

This matrix reveals the model's high accuracy in predicting passenger satisfaction. With 3,635 true negatives and 16,672 true positives, the model demonstrates strong predictive power, although there are minor misclassifications (152 in total), which are areas for further refinement.

**Logistic Regression Confusion Matrix** 

![](https://github.com/user-attachments/assets/eabc7eec-e29e-4228-bc9d-945a7e258196)


**Overall Performance Comparison**
![](https://github.com/user-attachments/assets/c1f2d106-90ee-429b-b67a-3ea7bdf0101c)

While regression models are often used for their simplicity and interpretability, they may not always capture complex, non-linear relationships within the data.

Random Forest excels in classification tasks by averaging multiple trees to improve accuracy and mitigate the risk of over-fitting, which is critical in complex datasets like ours. It handles both categorical and numerical data effectively, making it versatile for diverse features such as passenger demographics and service ratings. Additionally, it provides feature importance metrics, helping to identify which factors most influence satisfaction. Conversely, it can be computationally intensive, requiring significant processing power and memory, especially with large datasets. It also tends to produce less interpretable models compared to single decision trees, as the ensemble nature obscures individual decision paths

**Evaluation of the Model**

**Loss Function** The Random Forest model uses Gini, which measures the frequency at which a randomly chosen element would be incorrectly classified, providing a criterion for the quality of splits (Liaw & Wiener, 2019).
**Accuracy Metrics** 

![](https://github.com/user-attachments/assets/f579711b-44eb-431d-b7a1-0d5cfa1f439c)

The model achieved an impressive accuracy of 99%. Precision for the 'Satisfied' class (Y) is 0.99, indicating that 99% of the predictions for satisfied passengers were correct. Recall for the same class is 1.00, meaning the model correctly identified all satisfied passengers. The F1-score, the harmonic mean of precision and recall, is also high at 0.99, reflecting the model's balanced performance.


##  Conclusion
The findings indicate that customer satisfaction is influenced by multiple factors, including service quality, travel type, and demographic attributes. The use of machine learning models has provided a structured approach to predicting satisfaction levels with reasonable accuracy

## Recommendations

**Service Enhancements:** Focus on improving key service factors identified as major drivers of satisfaction.

**Targeted Marketing:** Utilize demographic insights to tailor customer engagement strategies.

**Model Optimization:** Further refine predictive models using advanced techniques for better accuracy.
