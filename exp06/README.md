# House Price Prediction  
It is a [2nd ML kaggle competition](https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr) with KaKR(Kaggle Korea).  
we can get 20 information about house (sqft_living, house grade, how many bathrooms etc.)   
we will predict house price!!  

In this repository, 3 files exist.
* [baseline](https://github.com/estela19/AIFFEL/blob/master/exp06/baseline.ipynb)  
* [EDA_practice](https://github.com/estela19/AIFFEL/blob/master/exp06/EDA_practice.ipynb)
* [randomgrid](https://github.com/estela19/AIFFEL/blob/master/exp06/randomgrid.ipynb) (this is submit file !!)  


### baseline  
It is provided by KaKR.  
with sample EDA and Model tuning.  

### EDA_practice  
It is my first file for EDA.  
I try to delete low co-relation colums, convert bathrooms from float to int, grouping sqft colums.  
But, it is hard to convert model input.  So, I give up 😂  

### randomgrid  
this is submit file !!  
It has simple EDA (see datas and just regulazation) and simple model.  
It use Xgboost model with random grid search for find hyper parameter.  


## Thinking about..
* How can we get useful data from raw data 
* How about using deep learning to regressing price