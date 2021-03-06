# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

------
## 1. Project Definition

### 1.1. Project Overview

Mail-order is the buying of goods or services by mail delivery. The buyer places an order for the desired products with the merchant through some remote method such as through a telephone call or web site. Then, the products are delivered to the address supplied by the customer.

Direct marketing is a type of advertising campaign that seeks to bring an action in a selected group of consumers (such as an order, visit the store or the website of the mark or a request for information) in response of a direct communication initiated by the marketer. Some of the benefits of direct marketing campaigns are that it helps a company increase the sales with current and former clients (through direct marketing a company can communicate directly with current customers to keep alive the relationship bringing value, but also get back in touch with old customers and generate new sales opportunities) and through direct contact with the customers the company can create new business opportunities by customizing its promotions, emails and offers to create an instant bond. 
  
In this project we will analyse demographics data for customers that form the core customer base for a mail-order sales company located in Germany and compare it to the general population. We will than apply unsupervised learning techniques in order to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. We will afterwards create a supervised model that will predict which individuals should be targeted for a particular direct marketing campaign in order to have the highest ROI (return on investment) for that campaign.

The data and outline of this project was provided by Arvato Financial Solutions, a Bertelsmann subsidiary.

### 1.2. Problem Statement

With traditional advertising methods, a company may be spending 1000 Euros for a billboard to advertise to 700 people in a day, with no control over demographic or being able to objectively measure the impact of this investment. Usually the costs linked to marketing campaigns, represent an important part of the yearly budget. One way to optimize this budget is to address online direct marketing to a specific audience, which allows a company to set realistic goals and improve the sales with a smaller budget. 

Even if a direct marketing campaign represents a smaller investment done by a company, when using a large database for a direct marketing campaign, a key objective is to identify and remove prospects not likely to respond, or that represent a high risk for the organisation. This optimisation of the direct marketing campaign tends to minimise the time and resources spent on leads that won't result in ROI.

In order to achieve results with only a small percentage of the cost of traditional advertising, a direct marketing campaign should be optimized by being properly directed, going through the removal of people not interested. We can achieve this through targeting the main segments of customers. This technique will increase the response rate and the campaign will become more profitable.

In order to optimize a direct marketing campaign, we will identify the segments that exist in the general population by using the KMean technique in order to create the segmentation of the population. 

Once we have identified the segments in the general population, we will predict in which of these segments our existing customers are located. We will than calculate the percentages in the general population and in the customers dataset that the given segment covers. All the segments for which the **percentage of customers is bigger than the percentage in the general population**, are our **target segments** as the people in these segments are more likely to convert and become a customer. We should than reach to this particular audience with personalized messages.

Once we have reached to the selected audience, we will than check the features of the people that have responded to a particular campaign and do a further filtering by using supervised learning and applying multiple stacked models in order to **predict the probability** of a given person to reply to the campaign.

We will also use the SHAP values in order to understand and explain our models� predictions. We will use the SHAP values at global level for the complete dataset in order to identify the importance of every feature, but also at local level for a couple of records in order to better understand the people which have already replied or not to the campaign.


### 1.3. Metrics

We will be using two different metrics depending on the problem we will be trying to solve.

#### Unsupervised model

The unsupervised model has to create the segmentation of our population.

We will be using AIC (Akaike information criterion) and BIC (Bayesian information criterion) in order to determine the optimal number of segments that we should keep in the model and do the comparison of different types of models.

The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given dataset. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection. 

The Bayesian information criterion (BIC) is based, in part, on the likelihood function and it is closely related to the AIC. 

The model with the lowest BIC and AIC is preferred.


#### Supervised model 

The supervised model has to predict if a person will reply or not to a given campaign, so we have a **binary classification** problem. 

When we check the distribution the two classes in the target, we can see that **only 1% of the total number of persons** to which we have reached out have responded, so we have a very **imbalanced target**. 

One metric which is insensitive to imbalanced classes is ROC - AUC (Receiver Operating Characteristic - Area Under the Curve). 

Also, we have chosen to use ROC - AUC metric as the stacked models will not predict if a given person will definitively reply or not to the campaign, but rather they will predict the **probability** that a given person will reply to the campaign.  


## 2. Analysis

### 2.1. Data Exploration

We have multiple files in CSV / XLS format associated with this project:
   - top-level list of attributes and descriptions, organized by informational category
   - detailed mapping of data values for each feature in alphabetical order
   - general population dataset - Demographics data for the general population of Germany
   - customer dataset - Demographics data for customers of a mail-order company
   - training dataset for the supervised model - Demographics data for individuals who were targets of a marketing campaign
   - testing dataset to generate predictions by using the supervised model - Demographics data for individuals who were targets of a marketing campaign

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighbourhood.

We start our data exploration by checking the number of rows and features that can be found in the general population dataset and in the customers dataset.

[numberOfRows]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/numberOfRows.png "Number of rows in the general population and the customers dataset"
![alt text][numberOfRows]

We also check all the distinct values for the different features in both datasets, and their distribution.

[columnValues]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/columnValues.jpg "columnValues.jpg"
![alt text][columnValues]

#### Analysis of categorical features

The analysis of categorical features consists in identifying the features that act as categories and should be considered as such.

In order to identify the features that are categorical, we use different methods:
   - we process the detailed mapping file provided in order to identify useful insights
   - we re-encode certain features to act as categorical features based on a custom mapping
   - we define a particular range of values for which the features having all values inside this given range and have the total number of unique values lower than a given threshold, are to be considered as categorical

By using the detailed mapping file provided, we can identify the following:
   - we have multiple features that use two or more encodings for the same meaning. We will check the definition dictionary and convert all multiple encodings in order to only keep the first one and re-encode all the values with the one we have decided to keep
   - we have multiple multi-level features, that we can split in single-level features

The following features are multi-level features:

   - CAMEO_DEUINTL_2015 - CAMEO classification 2015 - international typology. We create two new variables containing the family grouping and the wealth indicator.

[multiLevelFeatures_CAMEO_DEU]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_CAMEO_DEU.png "Multi-level features"
![alt text][multiLevelFeatures_CAMEO_DEU]

   - PRAEGENDE_JUGENDJAHRE - dominating movement in the person's youth (avantgarde or mainstream). We create four new variables containing the decade, the type of movement, the indication if the movement was Avantgarde / Mainstream, the location O / W / O+W.

[multiLevelFeatures_PRAEGENDE_JUGENDJAHRE]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_PRAEGENDE_JUGENDJAHRE.png "Multi-level features"
![alt text][multiLevelFeatures_PRAEGENDE_JUGENDJAHRE]

   - D19_BANKEN_DATUM - actuality of the last transaction for the segment banks TOTAL. We create two new variables containing the ACTIVITY_WITHIN_MONTHS and the Value_INCREASE.

[multiLevelFeatures_BANKEN_DATUM]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_BANKEN_DATUM.png "Multi-level features"
![alt text][multiLevelFeatures_BANKEN_DATUM]

   - D19_BANKEN_ONLINE_DATUM - actuality of the last transaction for the segment banks ONLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_BANKEN_OFFLINE_DATUM - actuality of the last transaction for the segment banks OFFLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_GESAMT_DATUM - actuality of the last transaction with the complete file TOTAL. Same mapping as D19_BANKEN_DATUM feature.

   - D19_GESAMT_ONLINE_DATUM - actuality of the last transaction with the complete file ONLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_GESAMT_OFFLINE_DATUM - actuality of the last transaction with the complete file OFFLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_TELKO_DATUM - actuality of the last transaction for the segment telecommunication TOTAL. Same mapping as D19_BANKEN_DATUM feature.

   - D19_TELKO_ONLINE_DATUM - actuality of the last transaction for the segment telecommunication ONLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_TELKO_OFFLINE_DATUM - actuality of the last transaction for the segment telecommunication OFFLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_VERSAND_DATUM - actuality of the last transaction for the segment mail-order TOTAL. Same mapping as D19_BANKEN_DATUM feature.

   - D19_VERSAND_ONLINE_DATUM - actuality of the last transaction for the segment mail-order ONLINE. Same mapping as D19_BANKEN_DATUM feature.

   - D19_VERSAND_OFFLINE_DATUM - actuality of the last transaction for the segment mail-order OFFLINE. Same mapping as D19_BANKEN_DATUM feature.

   - LP_LEBENSPHASE_GROB - life stage rough. We create three new variables containing the AGE, the FAMILY and the INCOME.

[multiLevelFeatures_LP_LEBENSPHASE_GROB]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_LP_LEBENSPHASE_GROB.png "Multi-level features"
![alt text][multiLevelFeatures_LP_LEBENSPHASE_GROB]

   - LP_LEBENSPHASE_FEIN - life stage fine. We create four new variables containing the AGE, the FAMILY, the INCOME and OTHER.

[multiLevelFeatures_LP_LEBENSPHASE_FEIN]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_LP_LEBENSPHASE_FEIN.png "Multi-level features"
![alt text][multiLevelFeatures_LP_LEBENSPHASE_FEIN]

   - LP_STATUS_FEIN - social status fine. We create two new variables for INCOME and OTHER.

[multiLevelFeatures_LP_STATUS_FEIN]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_LP_STATUS_FEIN.png "Multi-level features"
![alt text][multiLevelFeatures_LP_STATUS_FEIN]

   - LP_FAMILIE_FEIN - family type fine. We create three new variables containing the CHILD, the FAMILY and the GENERATIONAL split.

[multiLevelFeatures_LP_FAMILIE_FEIN]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_LP_FAMILIE_FEIN.png "Multi-level features"
![alt text][multiLevelFeatures_LP_FAMILIE_FEIN]

For certain categorical columns we have created a custom mapping that we apply in order to transform object values to numerical values. Below you can find an example for such a mapping:

[featureMapping_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/featureMapping_01.jpg "Features Mapping 01"
![alt text][featureMapping_01]

[featureMapping_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/featureMapping_02.jpg "Features Mapping 02"
![alt text][featureMapping_02]

Column 'EINGEFUEGT_AM' contains an encoding of type timestamp, so we will convert it to a timestamp and extract its components.

[EINGEFUEGT_AM]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/EINGEFUEGT_AM.png "EINGEFUEGT_AM"
![alt text][EINGEFUEGT_AM]

We consider that all the features that have all values between -2 and 10 and have less than 11 unique values are categorical features, so we will mark them as such. 


#### Analysis of Outliers


An outlier is an observation point that is distant from other observations.

We will be using the IQR (interquartile range) method in order to identify the outliers. We will apply a threshold of two times the IQR in order to define the minimum and maximum limits for acceptable values. 

Below you can find the list of features that have been identified as containing outliers:

[outliers_General]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_General.png "outliers_General"
![alt text][outliers_General]

When removing the outliers, we will take into consideration the type of the feature:
   - for categorical features we will use a mapping in which one value will be replaced by another value
   - for non-categorical features we will replace the values smaller or bigger than a maximum or minimum limit by the value of the limit

We will remap the following categorical features:
   - ALTERSKATEGORIE_GROB - age through first name analysis - all outliers are replaced

[outliers_ALTERSKATEGORIE_GROB]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_ALTERSKATEGORIE_GROB.png "outliers_ALTERSKATEGORIE_GROB"
![alt text][outliers_ALTERSKATEGORIE_GROB]

   - ARBEIT - share of unemployed person in the community - we check the definition file in order to decide if value 1 is to be considered as an outlier or not. The definition file indicates that this is a normal value, so we decide to keep it

[outliers_ARBEIT]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_ARBEIT.png "outliers_ARBEIT"
![alt text][outliers_ARBEIT]

   - KOMBIALTER - unknown description - all outliers are removed 

[outliers_KOMBIALTER]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_KOMBIALTER.png "outliers_KOMBIALTER"
![alt text][outliers_KOMBIALTER]

For column ALTER_KIND3 and ALTER_KIND4 we will just keep the information linked to the fact that the column is filled-in or not. 

For non-categorical features that have been identified as containing an outlier, we will execute the following operations:
   - if the outliers are smaller than Q1, we create a new column that flags the value as being smaller than the minimum threshold and we set the column value as being the threshold
   - if the outliers are bigger than Q3, we create a new column that flags the value as being bigger than the maximum threshold and we set the column value as being the threshold

The non-categorical features which are re-mapped based on the above rule are the following:

[outliers_05]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_05.png "outliers_05"
![alt text][outliers_05]

[outliers_06]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_06.png "outliers_06"
![alt text][outliers_06]

[outliers_07]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_07.png "outliers_07"
![alt text][outliers_07]

[outliers_08]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_08.png "outliers_08"
![alt text][outliers_08]

[outliers_09]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_09.png "outliers_09"
![alt text][outliers_09]

[outliers_10]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_10.png "outliers_10"
![alt text][outliers_10]

[outliers_11]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_11.png "outliers_11"
![alt text][outliers_11]


#### Analysis of MISSING / UNKNOWN values

The analysis of MISSING / UNKNOWN values consists in exploring the correlation between MISSING / UNKNOWN values and particular values for other features.

We start by calculating all the possible values and their distribution in both data frames.

Missing values could follow a specific pattern, so we take into account that we can have the following situations:
   - data is **Missing Completely at Random (MCAR)** - no relationship between the missingness of the data and any values, observed or missing (nothing systematic going on)
   - data is **Missing at Random (MAR)** - we have a systematic relationship between the propensity of missing values and the observed data, but not the missing data. Whether an observation is missing has nothing to do with the missing values, but it does have to do with the values of an individual�s observed variables (e.g.: women are less-likely to tell their age or weight)
   - data is **Missing Not at Random (MNAR)** - there is a relationship between the propensity of a value to be missing and its values. This is a case where the people with the lowest education are missing on education.

We check the pattern for MISSING values. Below you can see the first records which have the highest frequency:

[MISSING]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/MissingUnknown_MISSING.jpg "MISSING"
![alt text][MISSING]

We can see that as there are multiple cases where we can see a clear correlation between the number of missing values for the different features, so we can consider that the values are Missing Not at Random (MNAR).

We check the pattern for UNKNOWN values. Below you can see the first records which have the highest frequency:

[UNKNOWN]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/MissingUnknown_UNKNOWN.jpg "UNKNOWN"
![alt text][UNKNOWN]

We can see that as there are multiple cases where we can see a clear correlation between the number of unknown values for the different features, so we can consider that the values are Missing Not at Random (MNAR).

As we know that we are in a MNAR situation, we will execute a Principal component analysis (PCA) on both MISSING and UNKNOWN data in order to identify the missing / unknown flags with a reduced dimensionality.

[MissingUnknown_PCA_60_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/MissingUnknown_PCA_60_01.png "MissingUnknown_PCA_60_01"
![alt text][MissingUnknown_PCA_60_01]

[MissingUnknown_PCA_60_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/MissingUnknown_PCA_60_02.png "MissingUnknown_PCA_60_02"
![alt text][MissingUnknown_PCA_60_02]

The most important cases where values are unknown / missing together are the following:
   - case 1 when values are missing together
      - KBA05_KRSHERST2 - share of Volkswagen (referred to the county average)
      - KBA05_MAXAH     - most common age of car owners in the microcell
      - KBA05_MAXB      - most common age of the cars in the microcell
      - KBA05_MAXHERST  - most common car manufacturer in the microcell     
   - case 2 when values are missing together
      - D19_BANKEN_ONLINE_QUOTE_12  - amount of online transactions within all transactions in the segment bank  
      - D19_TELKO_ONLINE_QUOTE_12   - amount of online transactions within all transactions in the segment telecommunication     
      - D19_VERSAND_ONLINE_QUOTE_12 - amount of online transactions within all transactions in the segment mail-order       
      - D19_VERSI_ONLINE_QUOTE_12   - amount of online transactions within all transactions in the segment insurance     
   - case 3 when values are missing together 
      - LP_LEBENSPHASE_FEIN_Value_AGE     - life stage fine AGE 
      - LP_LEBENSPHASE_FEIN_Value_FAMILY  - life stage fine FAMILY 
      - LP_LEBENSPHASE_FEIN_Value_INCOME  - life stage fine INCOME 
      - LP_LEBENSPHASE_FEIN_Value_OTHER   - life stage fine OTHER 
   - case 1 when values are unknown together
      - KBA05_HERST3 - share of Ford/Opel
      - KBA05_KW2    - share of cars with an engine power between 60 and 119 KW
      - KBA05_KW3    - share of cars with an engine power of more than 119 KW 
      - KBA05_MAXAH  - most common age of car owners in the microcell     
   - case 1 when values are filled-in together  
      - D19_BANKEN_ONLINE_QUOTE_12  - amount of online transactions within all transactions in the segment bank  
      - D19_GESAMT_ONLINE_QUOTE_12  - amount of online transactions within all transactions in the complete file 
      - D19_LETZTER_KAUF_BRANCHE    - unknown description
      - D19_LOTTO                   - transactional activity based on the product group LOTTO

Based on the visualisation above, we will select **18 dimensions** for the PCA, as they capture almost 100% of the total variance for unknown and missing values. We do a retrain for the PCA with the selected number of components, and save the trained PCA.

[MissingUnknown_PCA_18]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/MissingUnknown_PCA_18.png "MissingUnknown_PCA_18"
![alt text][MissingUnknown_PCA_18]


#### Fill-in missing values

We decide to fill-in missing values in two cases:
   - dummy encode the object columns for which a custom mapping is not defined
   - fill in categorical features with -2

The features that still contain null values are the following:

[nullValues]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/nullValues.jpg "nullValues"
![alt text][nullValues]


#### Check for highly correlated features

We check for the highly correlated features as most algorithms are sensitive to them, and also, they don't bring any extra information.

[highlyCorrelated]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/highlyCorrelated.png "highlyCorrelated"
![alt text][highlyCorrelated]

We decide to drop the following columns: 
   - ANZ_STATISTISCHE_HAUSHALTE 
   - EINGEFUEGT_AM_DayOfYear 
   - EINGEFUEGT_AM_Minute 
   - EINGEFUEGT_AM_Second 
   - EINGEFUEGT_AM_Hour 
   - EINGEFUEGT_AM_Quarter 
   - EINGEFUEGT_AM_WeekOfYear 
   - GEBURTSJAHR 
   - KBA13_HERST_SONST 
   - KBA13_KMH_250 
   - ORTSGR_KLS9 
   - PLZ8_BAUMAX 
   - PLZ8_GBZ 
   - PLZ8_HHZ 
   - CAMEO_DEUINTL_2015_Value_1 
   - LP_LEBENSPHASE_FEIN_Value_FAMILY 
   - LP_LEBENSPHASE_FEIN_Value_INCOME 
   - LP_LEBENSPHASE_GROB_Value_FAMILY 
   - LP_FAMILIE_GROB 
   - PRAEGENDE_JUGENDJAHRE_Value_2 
   - _MIN_GEBAEUDEJAHR_MAX_1993 
   - D19_VERSAND_ONLINE_DATUM_Value_INCREASE


### 2.2. Algorithms and Techniques

#### Unsupervised modelling

Clustering is a method of unsupervised learning, where each datapoint or cluster is grouped to into a subset or a cluster, which contains similar kind of data points.

We will first use the PCA in order to reduce the dimensionality of our dataset and then we will create GMM (Gaussian Mixture Model) and KMean models. 

**A Gaussian mixture model** is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. 

The GMM is different from K-Means as it will not try to Hard assign data points to a cluster, but rather will use the probability of a sample to determine the feasibility of it belonging to a cluster.

**K-means clustering** is a type of unsupervised learning, which is used when you have unlabelled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
    - The centroids of the K clusters, which can be used to label new data
    - Labels for the training data (each data point is assigned to a single cluster)

#### Supervised modelling

In order to predict the probability of a person to reply to the mailing campaign we will create a **stack of LightGBM models** which will predict together this probability.

Every model will in fact be a stack of models trained through cross-validation and the AUC score will be the mean AUC score obtained by the individual models. 

We will start by searching the best hyperparameters for models using all available features by using a Bayesian search. 

Once we have a list of optimized hyperparameters, we will choose the first 10 and calculate the most important features. All these models will be trained on the exact same stratified splits.

We will also verify which are the most important 30 features.

The final stacking will be done on a combination of models trained with all the features and having the worst performers in the cross-validation dropped.


### 2.3. Benchmark

Model evaluation (including evaluating supervised and unsupervised learning models) is the process of objectively measuring how well machine learning models perform the specific tasks they were designed to do.

For the benchmarking of the implemented models we will use the following approach:
   - for unsupervised models, the model with the lower BIC will be considered as the best performer and the model with the highest BIC will be considered as the worst performer. The base BIC will the considered the BIC obtained by a GMM model using a TIED convergence type and fitter on two clusters. We will also check if particular relationships can be identified internally for the clusters. A good cluster will group common elements together, so we expect to have identifiable internal relations.
   - for supervised models, we will use the AUC scores in order to benchmark the performance of the models. A model with the highest AUC will be considered as the best performer. We will train a LightGBM model with default parameters and use it to predict the probabilities for the train and test datasets. The obtained AUC score will be our base score used for comparison.

## 3. Methodology

## 3.1. Data Pre-processing

Using the analysis and data exploration above, we execute the following pre-processing steps, by using the cleaning function (the same cleaning process will also be applied on the training and testing datasets):

[preprocessing_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/preprocessing_01.png "preprocessing_01"
![alt text][preprocessing_01]

[preprocessing_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/preprocessing_02.png "preprocessing_02"
![alt text][preprocessing_02]

For the unsupervised model we also execute the following supplementary steps:
   - we create custom scalers based on the general population values to be used for all datasets for which we will use the unsupervised model for predicting the clusters - these scalers will be saved as global variables to be used for scaling the identified features before predicting the clusters
   - we apply the scalers on the general population and the customers
   - we impute missing values for non-categorical features by using the mean value
   - we calculate the best number of reduced dimensions through PCA

We start by fitting a PCA on 200 dimensions (our dataset has 366 dimensions). You can find the results for the PCA below:

[preprocessing_03]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/preprocessing_03.png "preprocessing_03"
![alt text][preprocessing_03]

We also create a visualisation for the main dimensions which have been identified by the PCA and the link between the features.

[preprocessing_04]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/preprocessing_04.png "preprocessing_04"
![alt text][preprocessing_04]
   
Based on the results from the PCA fitted previously, we decide to keep the first 100 reduced dimensions.  

[preprocessing_05]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/preprocessing_05.png "preprocessing_05"
![alt text][preprocessing_05]


## 3.2. Implementation for unsupervised model

In order to create the clusters for our data, we start by fitting two GMM models, with covariance type TIED and FULL.

For these models we check the AIC and BIC results. In order to decide to use a GMM model, we must see that a global optimum is being found by the algorithm. In case no global optimum is found, then we can consider that the dataset is not a combination of gaussians distributions with unknown parameters and we should rather use a different type of model.

[cluster_GMM_AIC]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_GMM_AIC.png "cluster_GMM_AIC"
![alt text][cluster_GMM_AIC]

[cluster_GMM_BIC]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_GMM_BIC.png "cluster_GMM_BIC"
![alt text][cluster_GMM_BIC]

Based on the above visualizations we can see that the models could not find a global optimum, but rather only a local one.

We will than start fitting two KMean models, one fitted on 100 reduced dimensions, and another one on 150 reduced dimensions. We than compare their calculated BIC score with the BIC score of the most performant GMM model (covariance type FULL).

[cluster_KMean_BIC]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_BIC.png "cluster_KMean_BIC"
![alt text][cluster_KMean_BIC]

We see that the most performant model is a KMean fitted on 100 reduced dimensions. We check the inertia for the fitting on a range of clusters between 2 and 18.

[cluster_KMean_100_inertia]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_inertia.png "cluster_KMean_100_inertia"
![alt text][cluster_KMean_100_inertia]

Based on the above graphic we can see that the number of clusters should be either 4 or 5, so we calculate some detailed statistics on KMean models fitted on 3, 4 or 5 clusters in order to better understand how the labels for the different clusters will be distributed for the two datasets.

[cluster_KMean_100_heatmap]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_heatmap.png "cluster_KMean_100_heatmap"
![alt text][cluster_KMean_100_heatmap]

We can have the following insights:
   - when a KMean model is fitted for 3 clusters
      - for the general population - we have one big cluster that covers 53% of the observations
      - for the customers          - we have one big cluster that covers 46% of the observations  
      - for the customers          - we have the same distribution of the observations in two different clusters     
   - when a KMean model is fitted for 4 clusters
      - for the general population - we have one big cluster that covers 50% of the observations
      - for the customers          - we have one big cluster that covers 45% of the observations  
      - for the customers          - we have a similar distribution of the observations in two different clusters  
   - when a KMean model is fitted for 5 clusters
      - for the general population - we have more balanced clusters 
      - for the customers          - we have more balanced clusters  
      - for the customers          - we have three clusters that have a higher importance than in the general population. Cluster 0 (+15%), cluster 2 (+5%) and cluster 3 (+8%) should be the clusters targeted inside the general population for the marketing campaigns. The population inside the cluster 1 that has -16% contains a population not likely to convert into becoming a customer.

[cluster_KMean_100_barplot]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_barplot.png "cluster_KMean_100_barplot"
![alt text][cluster_KMean_100_barplot]

[cluster_KMean_100_3_dist]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_3_dist.png "cluster_KMean_100_3_dist"
![alt text][cluster_KMean_100_3_dist]    

[cluster_KMean_100_4_dist]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_4_dist.png "cluster_KMean_100_4_dist"
![alt text][cluster_KMean_100_4_dist]     

[cluster_KMean_100_5_dist]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_5_dist.png "cluster_KMean_100_5_dist"
![alt text][cluster_KMean_100_5_dist]

We also do a calculation of the optimum number of clusters by using the Elbow Method.

[cluster_KMean_100_elbow]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_elbow.png "cluster_KMean_100_elbow"
![alt text][cluster_KMean_100_elbow]

Based on the above calculation we decide to use a **KMean model** fitted on **100 reduced dimensions** with **5 clusters**.

We also decide to check the profiles for the most significative clusters for the customers.

For cluster 0 we can see that 20 reduced dimensions cover 90% of the variance.

[cluster_KMean_100_cluster_0_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_0_01.png "cluster_KMean_100_cluster_0_01"
![alt text][cluster_KMean_100_cluster_0_01]

[cluster_KMean_100_cluster_0_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_0_02.png "cluster_KMean_100_cluster_0_02"
![alt text][cluster_KMean_100_cluster_0_02] 

For cluster 2 we can see that 60 reduced dimensions cover 90% of the variance.

[cluster_KMean_100_cluster_2_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_2_01.png "cluster_KMean_100_cluster_2_01"
![alt text][cluster_KMean_100_cluster_2_01]

[cluster_KMean_100_cluster_2_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_2_02.png "cluster_KMean_100_cluster_2_02"
![alt text][cluster_KMean_100_cluster_2_02]  

For cluster 3 we can see that 60 reduced dimensions cover 90% of the variance.     

[cluster_KMean_100_cluster_3_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_3_01.png "cluster_KMean_100_cluster_3_01"
![alt text][cluster_KMean_100_cluster_3_01]

[cluster_KMean_100_cluster_3_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_100_cluster_3_02.png "cluster_KMean_100_cluster_3_02"
![alt text][cluster_KMean_100_cluster_3_02]


## 3.3. Implementation for supervised model

In order to implement the supervised model, we start by pre-processing the training dataset using the same cleaning function as above.

We will be using the unsupervised model fitted in the previous step in order to predict the cluster in which the observations are found.

The split in train and test datasets is done by specifying that the two datasets are to be stratified using the target and keep the same weight for the classes.

[supervised_balanced]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_balanced.png "supervised_balanced"
![alt text][supervised_balanced]

We than fit a LightGBM model with all features, without imputing the missing values, nor doing the scaling. Every model created is in fact a group of models trained within a 10 kfold cross-validation, from which the three less performant is dropped.

[supervised_base_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_base_01.png "supervised_base_01"
![alt text][supervised_base_01]

[supervised_base_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_base_02.png "supervised_base_02"
![alt text][supervised_base_02]

We also define a search hyperspace that we will be using in order to do a Bayesian parameter search. We define a search space for the following hyper-parameters:
   - number of estimators
   - boosting type
   - subsample
   - maximum depth
   - scale positive weight
   - number of leaves
   - learning rate
   - minimum child samples
   - alpha regularisation
   - lambda regularisation
   - column sample by tree
   
The most performant 150 models have the following parameters:

[supervised_search_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_search_02.png "supervised_search_02"
![alt text][supervised_search_02]       

[supervised_search_03]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_search_03.png "supervised_search_03"
![alt text][supervised_search_03]

[supervised_search_04]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_search_04.png "supervised_search_04"
![alt text][supervised_search_04]

The last step of the Bayesian search creates five stacking folders in which we copy the best performant 10, 20, 30, 60 or 100 models.

[supervised_search_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_search_01.png "supervised_search_01"
![alt text][supervised_search_01]

The best predictions on the test dataset are obtained with 10 stacked models. In the Kaggle competition, the best performer is the stack of 60 models.

[supervised_best_10_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_10_01.png "supervised_best_10_01"
![alt text][supervised_best_10_01]         

[supervised_best_10_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_10_02.png "supervised_best_10_02"
![alt text][supervised_best_10_02]  

[supervised_best_20_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_20_01.png "supervised_best_20_01"
![alt text][supervised_best_20_01]         

[supervised_best_20_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_20_02.png "supervised_best_20_02"
![alt text][supervised_best_20_02]  

[supervised_best_30_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_30_01.png "supervised_best_30_01"
![alt text][supervised_best_30_01]         

[supervised_best_30_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_best_30_02.png "supervised_best_30_02"
![alt text][supervised_best_30_02]

We also check which are the 30 most important features.   

[supervised_importance]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_importance.png "supervised_importance"
![alt text][supervised_importance]


## 4. Conclusion

For a direct marketing campaign it is very important to correctly identify the customers which will respond to a particular campaign.

By using unsupervised techniques we have identified that the general population can be splitted in five different clusters. When we check the inner distribution inside the clusters based on the customer type, product type and online buying, we can see that every cluster will rather contain a particular profile.

[cluster_KMean_inner]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/cluster_KMean_inner.png "cluster_KMean_inner"
![alt text][cluster_KMean_inner]

The supervised model can predict the probability that a customer will reply to a particular marketing campaing. By using SHAP values at global and local level we can identify the particularity of every customer.

[supervised_SHAP_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_SHAP_01.png "supervised_SHAP_01"
![alt text][supervised_SHAP_01]

[supervised_SHAP_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/supervised_SHAP_02.png "supervised_SHAP_02"
![alt text][supervised_SHAP_02]

