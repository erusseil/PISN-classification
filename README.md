# PISN-classification
Machine learning indentification of PISN among other objects in the Plasticc data set 

Hello user ! 

The Large Synoptic Survey Telescope (LSST) will soon be operational and will provide us with an inimaginable amount of data. To be precise, it will scan the sky, looking for objects with variable luminosity over time. Since analysing all the light curves manually is litteraly impossible, we need to figure a way to filter the data.

What if we want to predict the type of object we are looking at only using it's light curve ? For this we need to train a machine learning algorithm with a lot of data. Fortunately a simulation of the data that LSST could produce in 3 years exists : it is called PLAsTiCC.

It was original created for a Kaggle challenge in 2018. More here details here : https://www.kaggle.com/c/PLAsTiCC-2018/overview

The training sample was composed of about 30 000 objects and the testing sample was composed of about 3 500 000 objects. The difficulty comes from this huge difference of size between the samples. 
Since we expect LSST to discover theoretical that we've never observed before, some were added in the testing sample (only !). Algorithms were expected to classify correctly objects that were in both samples and to classify as 'unknown' objects that were only in the testing sample. In reality unseen objects were not classify as "unknown", and those exotic objects could slip between our hands if we are not capable of classifying them !

It is in this context that this github takes place. We will focus on identifying Pair Instability Supernovae (called PISN) in particular. Now that the testing sample is public, we will use the PISN data to train directly a machine learning model.

To begin with, you need to have the PLAsTiCC data set. It is downladable here : https://zenodo.org/record/2539456#.YED0lP4o9hE
For both testing and training you will find a data and a metadata file. The data file contains the light curves themselves, while metadata contains additionnal informations about the objects.
The training is made of 2 files : "plasticc_train_lightcurves.csv" and "plasticc_train_metadata.csv" 
The testing is made of 12 files : 11 "plasticc_test_lightcurves_xx.csv" and "plasticc_test_metadata.csv"

In addition I used the data of all PISN as separate files here : 




