# Final_Project_ML
Machine Learning (6350) Final Project

There are two types of files found in the final project repository, those that were used for pre-processing and those that take the pre-processed data and apply machine learning techniques on the processed data. All files that start with "Kaggle_" are those that take the pre-processed data and apply various algorithms to make predictions on the dataset. All other files: "ID3_Fill", "Data_Enc", and "Pre_Processing_Analysis", are used as pre-processing tools. "ID3_Fill" is used to fill in missing features using decision trees, and "Data_Enc" encodes all the categorical features into numerical features and concatenates all the data such that the output of this script is the final form data for the various machine learning algorithms. "Pre_Processing_Analysis" is a simple script used to view the missing fefatures and overall information of the dataset before pre-processing was started. 

The Kaggle Algorithm Scripts call the variable data that is finalized in "Data_Enc". Each Kaggle script uses Sklearn with the exception of the Neural Network script. 
