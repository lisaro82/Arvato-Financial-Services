# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services

------
## 1. Project Definition

### 1.1. Project Overview

Mail-order is the buying of goods or services by mail delivery. The buyer places an order for the desired products with the merchant through some remote method such as through a telephone call or web site. Then, the products are delivered to the address supplied by the customer.

Direct marketing is a type of advertising campaign that seeks to bring an action in a selected group of consumers (such as an order, visit the store or the website of the mark or a request for information) in response of a direct communication initiated by the marketer. Some of the benefits of direct marketing campains are that it helps a company increase the sales with current and former clients (through direct marketing a company can communicate directly with current customers to keep alive the relationship bringing value, but also get back in touch with old customers and generate new sales opportunities) and through direct contact with the customers the company can create new business opportunities by customizing its promotions, emails and offers to create an instant bond. 
  
In this project we will analyze demographics data for customers that form the core customer base for a mail-order sales company located in Germany and compare it to the general population. We will than apply unsupervised learning techniques in order to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company, and afterwards create a supervised model that will predict which individuals should be targeted during direct marketing campaigns in order to have the highest ROI (return on investment).

The data and outline of this project was provided by Arvato Financial Solutions, a Bertelsmann subsidiary.

### 1.2. Problem Statement

When 




    Take the segmentation and targeting. One of the great advantages of this type of marketing is that you can reach your specific audience segments with personalized messages. If you want to succeed, you should invest time in research to identify consumers most likely to convert and thus direct your efforts to actions that really work.


    Optimize your marketing budget. Addressing online direct marketing to a specific audience allows you to set realistic goals and improve your sales on a tight budget. If you optimize and properly direct your campaign, you will achieve results with only a small percentage of the cost of traditional advertising.




With direct marketing, companies are likely to have a benchmark success rate from previous similar campaigns that they can base their goals on. Companies can try to optimise it through various strategies, whether by trying to offer the best deals or by using behavioural tactics.

One of the most successful ways to **optimise direct marketing** results is through machine learning.

When using a large database for a direct marketing campaign, a key objective is to identify and remove prospects not likely to respond, or that represent a high risk to your organisation, to minimise time and resources spent on leads that won't result in ROI. By removing those not interested, you will improve your response rates: making your campaign more profitable.

---------
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:

    Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?
    Have you thoroughly discussed how you will attempt to solve the problem?
    Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?

### 1.3. Metrics

An essential aspect of direct marketing is that the consumer response is measurable.

---------
Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

Metrics

In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:

    Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?
    Have you provided reasonable justification for the metrics chosen based on the problem and solution?


## 2. Analysis

### 2.1. Data Exploration

---------
Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:

    If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?
    If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?
    If a dataset is not present for this problem, has discussion been made about the input space or input data for your problem?
    Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)


### 2.2. Data Visualization

---------
Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.

In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:

    Have you visualized a relevant characteristic or feature about the dataset or input data?
    Is the visualization thoroughly analyzed and discussed?
    If a plot is provided, are the axes, title, and datum clearly defined?


### 2.3. Algorithms and Techniques


---------
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:

    Are the algorithms you will use, including any default variables/parameters in the project clearly defined?
    Are the techniques to be used thoroughly discussed and justified?
    Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?
    

### 2.4. Benchmark


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
The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:

    Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?
    Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?
    Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?
    Can results found from the model be trusted?

## 4.2. Justification

---------
The final results are discussed in detail.
Exploration as to why some techniques worked better than others, or how improvements were made are documented.

In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:

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

    Does the project report you’ve written follow a well-organized structure similar to that of the project template?
    Is each section (particularly Analysis and Methodology) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
    Would the intended audience of your project be able to understand your analysis, methods, and results?
    Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
    Are all the resources used for this project correctly cited and referenced?
    Is the code that implements your solution easily readable and properly commented?
    Does the code execute without error and produce results similar to those reported?

