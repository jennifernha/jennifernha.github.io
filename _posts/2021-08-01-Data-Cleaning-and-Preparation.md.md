---
title: "Data Cleaning and Preparation"
date: 2021-05-28 02:49:28 -0400
categories: data cleaning, data preparation
---

Data Cleaning & Preparation

Unlike the past, it has become very convenient and easy to download large datasets with the internet. There are many free resources available online, including data on everything from business to finance, entertainment, and government. 

What follows with this benefit is that there is often too much information available, which requires data cleaning and preparation when using these datasets. I, an aspiring data scientist, also went through these when I worked on my Microsoft Movie Analysis and King County House Sales projects. Therefore, I’d like to dedicate this blog post to explaining the essential steps for data cleaning and preparation.

1. Get summary of the dataframe
After importing the dataset you will be working with in the Jupyter Notebook, the very first step is to get a concise summary of the dataframe. Using the .info() method, take a look at the non-null count and data type for each column with a total number of entries.

2. Drop unnecessary column(s)
Then, remove the columns that you don’t need for your research, analysis, project, etc. In most cases, the dataset you work with will be large so we would like to keep things simple. You can remove the column(s) that you don’t need with the .drop() method.

3. Check/Convert Data Type
Now that you only have what you need, you want to make sure each column has the correct data type. You should have gotten the data type information when you used the .info() method at the very beginning. We want to make sure that texts show as string, and numbers as integer or float to avoid running into ValueError. If the stored data type is incorrect, you can easily make the change using the .astype() method. 

4. Missing Values
From step 1, you should also have the number of non-null counts in each column. With this information, you can easily calculate the number of null values within the dataset. Using the .isna.sum() method, you can also get the number of null values in each column in one view. If you are taking a look at a specific column(s), you can also use .value_counts(normalize = True) to return null and non-null values in binomial. 

After getting this information, you must decide whether you should drop those null values or replace them with 0, using .fillna(0, inplace = True). Use your best judgement here as missing values might be important case by case. 

5. Duplicates
Lastly, you should take a look at the duplicate values. You can use .duplicated().sum() for specific column(s) to get the total number of duplicate values. Similar to the missing values, not all duplicate values need to be removed from the dataset. Depending on how you are using the data and what the data tells you, you should decide whether you are going to keep the duplicate values or not. 

Regardless of which dataset you work with, there is no perfect dataset that doesn’t require data cleaning and preparation process. You want to work with the “cleanest” dataset as much as possible. I hope this blog post can help many other aspiring data scientists!

