# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

------
## 1. Project Definition

### 1.1. Project Overview

Mail-order is the buying of goods or services by mail delivery. The buyer places an order for the desired products with the merchant through some remote method such as through a telephone call or web site. Then, the products are delivered to the address supplied by the customer.

Direct marketing is a type of advertising campaign that seeks to bring an action in a selected group of consumers (such as an order, visit the store or the website of the mark or a request for information) in response of a direct communication initiated by the marketer. Some of the benefits of direct marketing campains are that it helps a company increase the sales with current and former clients (through direct marketing a company can communicate directly with current customers to keep alive the relationship bringing value, but also get back in touch with old customers and generate new sales opportunities) and through direct contact with the customers the company can create new business opportunities by customizing its promotions, emails and offers to create an instant bond. 
  
In this project we will analyze demographics data for customers that form the core customer base for a mail-order sales company located in Germany and compare it to the general population. We will than apply unsupervised learning techniques in order to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company, and afterwards create a supervised model that will predict which individuals should be targeted during direct marketing campaigns in order to have the highest ROI (return on investment).

The data and outline of this project was provided by Arvato Financial Solutions, a Bertelsmann subsidiary.

### 1.2. Problem Statement

With traditional advertising methods, a company may be spending 1000 Euros for a billboard to advertise to 700 people in a day, with no control over demographic or being able to objectivly measure the impact of this investment. Usually the costs linked to marketing campaigns, represent an important part of the yearly budget. One way to optimize this budget is to address online direct marketing to a specific audience, which allows a company to set realistic goals and improve the sales with a smaller budget. 

Even if a direct marketing campaign represents a smaller investment done by a company, when using a large database for a direct marketing campaign, a key objective is to identify and remove prospects not likely to respond, or that represent a high risk to the organisation. This optimisation of the direct marketing campaign tends to minimise the time and resources spent on leads that won't result in ROI. By removing those not interested, the company will improve its response rates: making the campaign more profitable by being properly directed.

In order to achieve results with only a small percentage of the cost of traditional advertising, a direct marketing campaign should be optimezed and properly directed.

In order to optimize a direct marketing campaign, we will identify the segments that exist in the general population by using the GMM (Gaussian Mixture Model) technique in order to create the segmentation of the population. 

Once we have identified the segments in the general population, we will predict in which of these segments our existing customers are located. We will than calculate the percentages in the general population and in the customers dataset that the given segment covers. All the segments for which the **percentage of customers is bigger than the percentage in the general population**, are our **target segments** as the people in these segments are more likely to convert and become a customer. We should than reach to this particular audience with personalized messages.

Once we have reached to the selected audience we will than check the features of the people that have responded to the campain and do a further filtering by using supervised learning and applying multiple stacked models in order to **predict the probability** of a given person to reply to the campaign.

We will also use the SHAP values in order to understand and explain our models prediction. We will use the SHAP values at global level for the complete dataset in order to identify the importance of every feature, but also at local level for a couple of records in order to better understand the people which have already replied or not to the campain.


### 1.3. Metrics

We will be using two different metrics depending on the problem we will be trying to solve.

#### Unsupervised model

The unsupervised model has to create the segmentation of our population.

We will be using AIC (Akaike information criterion) and BIC (Bayesian information criterion) in order to determine the optimal number of segments that we should keep in the model. 

We will also check that the algorithm has correctly converted. 

The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a given dataset. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection. 

The Bayesian information criterion (BIC) is based, in part, on the likelihood function and it is closely related to the AIC. 

The model with the lowest BIC and AIC is preferred.


#### Supervised model 

The supervised model has to predict if a person will reply or not to a given campain, so we have a binary classification problem. 

When we check the distribution the the two classes in the target we can see that only 1% of the total number of persons to which we have reached out have responded, so we have a very imbalanced target. 

One metric which is insensitive to imbalanced classes is ROC - AUC (Receiver Operating Characteristic - Area Under the Curve). 

Also, we have choosen to use ROC - AUC metric as the stacked models will not predict if a given person will definetively reply or not to the campain, but rather they will predict the probability that a given person will reply to the campaign.  


## 2. Analysis

### 2.1. Data Exploration

We have multiple files in CSV / XLS format associated with this project:
   - top-level list of attributes and descriptions, organized by informational category
   - detailed mapping of data values for each feature in alphabetical order
   - general population dataset - Demographics data for the general population of Germany
   - customer dataset - Demographics data for customers of a mail-order company
   - training dataset for the supervised model - Demographics data for individuals who were targets of a marketing campaign
   - testing dataset to generate predictions by using the supervised model - Demographics data for individuals who were targets of a marketing campaign

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood.

We start our data exploration by checking the number of rows and features that can be found in the general population dataset and in the customers dataset.

[numberOfRows]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/numberOfRows.png "Number of rows in the general population and the customers dataset"
![alt text][numberOfRows]

#### Analysis of categorical features

The analysis of categorical features consists in identifying the features that act as categories and should be considered as such.

In order to identify the features that are categorical, we use different methods:
   - we process the detailed mapping file provided in order to identify usefull insights
   - we re-encode certain features to act as categorical features based on a custom mapping
   - we define a particular range of values for which all features having the values inside the given range are to be considered as categorical

By using the detailed mapping file provided, we can identify the following:
   - we have multiple features that use two or more encodings for the same meaning. We will check the definition dictionary and convert all multiple encodings in order to only keep the first one.
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

   - LP_LEBENSPHASE_GROB - lifestage rough. We create three new variables containing the AGE, the FAMILY and the INCOME.

[multiLevelFeatures_LP_LEBENSPHASE_GROB]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/multiLevelFeatures_LP_LEBENSPHASE_GROB.png "Multi-level features"
![alt text][multiLevelFeatures_LP_LEBENSPHASE_GROB]

   - LP_LEBENSPHASE_FEIN - lifestage fine. We create four new variables containing the AGE, the FAMILY, the INCOME and OTHER.

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

We consider that all the features that have all values between -2 and 10 are categorical features, so we will mark them as such. 


#### Analysis of Outliers


An outlier is an observation point that is distant from other observations.

We will be using the IQR (interquartile range) method in order to identify the outliers. We will apply a threshold of two times the IQR in order to define the minimum and maximum limits for acceptable values.

[outliers_01]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_01.png "outliers_01"
![alt text][outliers_01]

We will remap the following categorical features:
   - ALTERSKATEGORIE_GROB - age through first name analysis

[outliers_02]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_02.png "outliers_02"
![alt text][outliers_02]

   - ARBEIT - share of unemployed person in the community

[outliers_03]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_03.png "outliers_03"
![alt text][outliers_03]

   - KOMBIALTER - unknown description

[outliers_04]: https://github.com/lisaro82/Arvato-Financial-Services/blob/master/screenShots/outliers_04.png "outliers_04"
![alt text][outliers_04]


For column ALTER_KIND3 and ALTER_KIND4 we will just keep the information linked to the fact that the column is filled-in or not. 

For the columns which were identified as having outliers, and are not identified as being categorical, we do the following operations:
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

We start by calculating all the possible values and their distribution in both dataframes.

Missing values could follow a specific pattern, so we take into account that we can have the following situations:
   - data is **Missing Completely at Random (MCAR)** - no relationship between the missingness of the data and any values, observed or missing (nothing systematic going on)
   - data is **Missing at Random (MAR)** - we have a systematic relationship between the propensity of missing values and the observed data, but not the missing data. Whether an observation is missing has nothing to do with the missing values, but it does have to do with the values of an individual’s observed variables (eg: women are less-likely to tell their age or weight)
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

Based of the visualisation above, we will select **18 dimensions** for the PCA, as they capture almost 100% of the total variance for unknown and missing values. We do a retrain for the PCA with the selected number of components, and save the trained PCA.

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

We check for the highly correlated features as most algorithms are sensitive to them, and also they don't bring any extra information.

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

#### Unsupervised modeling

Clustering is a method of unsupervised learning, where each datapoint or cluster is grouped to into a subset or a cluster, which contains similar kind of data points.

We will first use the PCA in order to reduce the dimensionamity of our dataset and than we will create a GMM (Gaussian Mixture Model) model. 

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The GMM is different from K-Means as it will not try to Hard assign data points to a cluster, but rather will use the probability of a sample to determine the feasibility of it belonging to a cluster.

The main avantage for using this type of clustering on our dataset is that it will not bias the cluster sizes to have specific structures as does K-Means (Circular).


#### Supervised modeling

In order to predict the probability of a person to reply to the mailing campaign we will create a stack of LightGBM models which will predict together this probability.

Every model will in fact be a stack of models trained through cross-validation and the AUC score will be the mean AUC score obtained by the individual models. 

We will start by searching the best hyperparamaters for models using all available features by using a Bayesian search. 

Once we have a list of optimized hyperparamaters, we will choose the first 10 and calculate the most important features. All these models will be trained on the exact same stratified splits.

We than choose the first 30 features and do a new search for the optimal parameters.

The final stacking will be done on a combination of models trained with all the features and with only the first 30 more important features.


### 2.3. Benchmark

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
---------
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:

    Has some result or value been provided that acts as a benchmark for measuring performance?
    Is it clear how this result or value was obtained (whether by data or by hypothesis)?



## 3. Methodology

(approx. 3-5 pages)
## 3.1. Data Preprocessing

---------
All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:

    If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?
    Based on the Data Exploration section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?
    If no preprocessing is needed, has it been made clear why?

## 3.2. Implementation

---------
The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:

    Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?
    Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?
    Was there any part of the coding process (e.g., writing complicated functions) that should be documented?

## 3.3. Refinement

---------
The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:

    Has an initial solution been found and clearly reported?
    Is the process of improvement clearly documented, such as what techniques were used?
    Are intermediate and final solutions clearly reported as the process is improved?


## 4. Results

(approx. 2-3 pages)
## 4.1. Model Evaluation and Validation

---------
The final modelâ??s qualities â?? such as parameters â?? are evaluated in detail. Some type of analysis is used to validate the robustness of the modelâ??s solution.

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the modelâ??s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:

    Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?
    Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?
    Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?
    Can results found from the model be trusted?

## 4.2. Justification

---------
The final results are discussed in detail.
Exploration as to why some techniques worked better than others, or how improvements were made are documented.

In this section, your modelâ??s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:

    Are the final results found stronger than the benchmark result reported earlier?
    Have you thoroughly analyzed and discussed the final solution?
    Is the final solution significant enough to have solved the problem?

## 5. Conclusion

(approx. 1-2 pages)
## 5.1. Free-Form Visualization

---------
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:

    Have you visualized a relevant or important quality about the problem, dataset, input data, or results?
    Is the visualization thoroughly analyzed and discussed?
    If a plot is provided, are the axes, title, and datum clearly defined?

## 5.2. Reflection

---------
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:

    Have you thoroughly summarized the entire process you used for this project?
    Were there any interesting aspects of the project?
    Were there any difficult aspects of the project?
    Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?

## 5.3. Improvement

---------
Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.

In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:

    Are there further improvements that could be made on the algorithms or techniques you used in this project?
    Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?
    If you used your final solution as the new benchmark, do you think an even better solution exists?



------------------------------------------------------------------------
Before submitting, ask yourself. . .

    Does the project report youâ??ve written follow a well-organized structure similar to that of the project template?
    Is each section (particularly Analysis and Methodology) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
    Would the intended audience of your project be able to understand your analysis, methods, and results?
    Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
    Are all the resources used for this project correctly cited and referenced?
    Is the code that implements your solution easily readable and properly commented?
    Does the code execute without error and produce results similar to those reported?

