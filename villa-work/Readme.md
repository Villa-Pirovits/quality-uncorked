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
 
* Aquire data from database
 
* Prepare data
   * Create Engineered columns from existing data
       * baseline
       * rating_difference
       * game_rating
       * lower_rated_white
       * time_control_group
 
* Explore data in search of drivers for churn
   * Answer the following initial questions
       * Are customers with DSL more or less likely to churn?
       * What month are customers most likely to churn and does that depend on their contract type?
       * Is there a service that is associated with more churn than expected?
       * Do customers who churn have a higher average monthly spend than those who don't?
      
* Develop a Model to predict if a customer will churn
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best clusters based on data run on model
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary


:
Fixed acidity in wine refers to the  These acids contribute to the tartness, sourness, and overall acidity of the wine. Examples of fixed acids in wine include tartaric acid, malic acid, and citric acid. The level of fixed acidity affects the balance and perceived freshness of the wine.

:
Volatile acidity refers to the concentration of volatile acids in wine, primarily acetic acid. It is responsible for the vinegar-like smell and taste that can occur in wines with high levels of volatile acidity. In small amounts, volatile acidity can contribute to complexity and aroma, but excessive levels can be a wine fault.

:
Citric acid is a type of fixed acid found in wine. It adds a citrusy and refreshing flavor to the wine, enhancing its overall freshness and acidity.

:
Residual sugar refers to the amount of sugar that remains in the wine after fermentation. It is measured in grams per liter (g/L). Wines with higher residual sugar levels will taste sweeter, while wines with very low or no residual sugar are considered dry.

:
Chlorides are salts of chlorine that can be found in wine. In small amounts, chlorides can contribute to the wine's overall flavor and mouthfeel. However, excessive levels can negatively impact the taste and balance of the wine.

:
Free sulfur dioxide (SO2) is a preservative commonly added to wines to prevent oxidation and microbial spoilage. It also acts as an antioxidant. The level of free SO2 in wine is regulated to ensure proper preservation without negatively affecting taste and aroma.

:
Total sulfur dioxide refers to the sum of both free and bound forms of SO2 in wine. Bound SO2 combines with other compounds and is less effective as a preservative. Winemakers must carefully manage the total SO2 levels to maintain wine quality and stability.

:
As mentioned earlier, density in wine refers to its mass per unit volume (g/mL or kg/L). It can provide insights into the wine's composition and concentration.

:
pH is a measure of acidity or alkalinity in wine on a logarithmic scale from 0 to 14. Lower pH values indicate higher acidity, while higher values indicate lower acidity. The pH level influences the stability and microbial activity in the wine.

 (Sulphites):
Sulphates, or sulfites, are sulfur-containing compounds used as preservatives in winemaking. They prevent unwanted oxidation and microbial spoilage, ensuring wine quality and shelf life.

:
Alcohol content in wine is expressed as a percentage of ethanol by volume. It is a key determinant of a wine's body, warmth, and perceived intensity. Higher alcohol content can contribute to a wine's richness and mouthfeel.








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


# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from 
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Customers with Fiber Optics have more churn than DSL.
* Encouraging customers to be on automatic payment plan will seems to reduce churn.
* 643 manual check writers churned which is a 45% churn rate for all payment types.
* When the monthly charges reached approximate \$70 the churn rate rised.
* The median monthly payment for customers who churns is \$79.70
* Customers who do not churn makeup 73% of the data
* The final model failed to significantly outperform the baseline.
* Possible reasons include:
    “payment_type” and “contract_type” may not have had meaningful relation to who will churn.
    Since monthly charges" seems to be a larger contributor to churn, adding more of the services to see which service may be contributing to churn. 
 
# Recommendations
* This may be simple enough but have a column for reason for caneling service. Helpful to pinpoint issues and improve service.


# Next Steps
* Explore the relation of Fiber Optics to churn. Services like tech support or streaming services could also be explored.
