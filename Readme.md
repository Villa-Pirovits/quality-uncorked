# Quality Ucorked
 
# Project Description
 
We have been tasked to find drivers of wine quality for the California Wine Institute. They are most interested in seeing how utilizing clusters will affect a machine learning model. We will deliver a slide deck presentation to share your finding
 
# Project Goal
 
* Discover drivers that cause customer to churn
* Use drivers to develop a machine learning model to classify the customer as churn or not. 
* Predict the quality of wine while incorporating unsupervised learning techniques.

 
# Initial Thoughts
 
My initial hypothesis is that drivers of rice.
 
# The Plan
 
* Aquire data from https://data.world/food/wine-quality
 
* Prepare data  
   * Create Engineered columns from cluster model
       * cluster_2
       * cluster_3
       * cluster_4
 
* Explore data in search of drivers for quality
   * Answer the following initial questions
       * Density, how does that affect residual sugars?
       * How does alocohol affect density?
       * Is the average pH higher in red or white wine?
       * Is the average alcohol higher in red or white wine?
      
* Develop a Model to predict the quality score of wine
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best clusters based on data run on model
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|fixed acidity| total concentration of non-volatile acids present.|
|volatile acidity| the concentration of volatile acids in wine|
|citric acid| fixed acid found in wine|
|residual sugar| the amount of sugar that remains in the wine after fermentation|
|chlorides| salts of chlorine that can be found in wine|
|free sulfur dioxide| preservative commonly added to wines to prevent oxidation and microbial spoilage|
|total sulfur dioxide| the sum of both free and bound forms of SO2 in wine|
|density| mass per unit volume (g/mL or kg/L)|
|pH| a measure of acidity or alkalinity in wine on a logarithmic scale from 0 to 14|
|sulphates| sulfur-containing compounds used as preservatives in winemaking|
|alcohol| percentage of ethanol by volume|
|quality| score of wine, ranging from 3-9|
|type| type of wine, red or white|

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from https://data.world/food/wine-quality
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* ---
* ---
* ---
* ---
* ---
* ---
* The final model failed to significantly outperform the baseline.
* Possible reasons include:
    “payment_type” and “contract_type” may not have had meaningful relation to who will churn.
    Since monthly charges" seems to be a larger contributor to churn, adding more of the services to see which service may be contributing to churn. 
 
# Recommendations
* This may be simple enough but have a column for reason for caneling service. Helpful to pinpoint issues and improve service.


# Next Steps
* Explore the relation of Fiber Optics to churn. Services like tech support or streaming services could also be explored.
