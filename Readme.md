# Quality Ucorked
 
# Project Description
 
We have been tasked to find drivers of wine quality for the California Wine Institute. They are most interested in seeing how utilizing clusters will affect a machine learning model. We will deliver a slide deck presentation to share your finding
 
# Project Goal
 
* Discover drivers that predict wine quality
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
* Clustering based on 2 features (density and alcohol), 3 features (residual sugar, total sulfur dioxide, alcohol), and 4 features (volatile acidity, chlorides, density, alcohol) with k=4 for each produced divergent clusters that performed nearly identically in the final regression models.
* An ANOVA test was performed to show that the clustered groups were in fact different and valuable 
* We kept the clusters in new dummy columns and used them as features to perform Logistic and Polynomial Regression on.
* Our models all performed around with an RMSE around .72 for the Train data, .73 for the validate data.
* Our test RMSE using 2 clusters came out at .75 units of quality
* We beat our baseline based on mean of .87 units by .12 points.
* The final model marginally outperformed baseline.

 
# Recommendations
* In a future model, consider dropping or combining features through feature engineering in order to reduce multicolinearity issues.  For example total sulfur dioxide and free sulfur dioxide do not both need to be in the model.  Also Alcohol and density are highly correlated as is residual sugar.  
* Reducing the total number of columns may help in creating a better linear regression.
* Classification also might provide a better result.

# Next Steps
* Next steps include reevaluating the efficacy of the clustering all together.  If it shows a better result with different features, then we can hone in and narrow our focus with feature engineering where we combine correlated features and drop columns that do not add to our model.  
* Perhaps also there are differing determinants of quality between red and white wine and "manually clustering" the two might be an interesting to explore what features are in fact different between the two populations and if that has a differential impact of quality vs predicted quality.