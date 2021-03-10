# PISN-classification

Hello user ! 

The Large Synoptic Survey Telescope (LSST) will soon be operational and will provide us with an inimaginable amount of data. To be more specific, it will scan the sky, looking for objects with variable luminosity over time. Since analysing all the light curves manually is litteraly impossible, we need to figure a way to filter the data.

What if we want to predict the type of object that we are looking at only using it's light curve ? For this we need to train a machine learning algorithm with a lot of data. Fortunately a simulation of the data that LSST could produce in 3 years exists : it is called PLAsTiCC.

It was original created for a Kaggle challenge in 2018. More here details here : https://www.kaggle.com/c/PLAsTiCC-2018/overview

The training sample was composed of about 30 000 objects and the testing sample was composed of about 3 500 000 objects. The difficulty comes from this huge difference of size between the samples. 
Since we expect LSST to discover theoretical objects that we've never observed before, some were added in the testing sample (only !). Algorithms were expected to classify correctly objects that were in both samples and to classify as 'unknown' objects that were only in the testing sample. In reality unseen objects were not classify as "unknown", and those exotic objects could slip between our hands if we are not capable of classifying them !

It is in this context that this github takes place. We will focus on identifying Pair Instability Supernovae (called PISN) in particular. Now that the testing sample is public, we will use the PISN data to train directly a machine learning model.

To begin with, you need to have the PLAsTiCC data set. It is downladable here : https://zenodo.org/record/2539456#.YED0lP4o9hE

For both testing and training you will find a data and a metadata file. The data file contains the light curves themselves, while metadata contains additionnal informations about the objects.
The training is made of 2 files : "plasticc_train_lightcurves.csv" and "plasticc_train_metadata.csv" 
The testing is made of 12 files : 11 "plasticc_test_lightcurves_xx.csv" and "plasticc_test_metadata.csv"

In addition I used the data of all PISN as separate files here :

In the folder FilterDataBase you can find a notebook 'FusePISN.ipynb' that fuses all PISN files into one dataframe.

Once all files are downloaded we are ready. We can distinguish three main step : Filter ; Parametrise ; Predict

## Filter dataset

The first thing we usually do with light curves is to translate the time to 0. This mean that we set the time of the first point to be 0, and we apply the same translation to all the points on the curve. Also we might want to use only specific light curves and this is why original dataset needs to be transformed before using it. In the folder FilterDataBase there is the script "data_base.py" that allows that . It contains a functions 'create' that will return you a clean filtered dataset. At each step informations about the filters applied and remaining objects are printed. All those informations are saved in a txt file.


def create(data,metadata,band_used,name,PISNdf='',ratioPISN=-1,training=True,dff=True,extra=True,Dbool=False,complete=True,mini=5,norm=True):
 
 
 
 data : the light curve data frame
 
 metadata : the corresponding meta data frame
 
 PISNfile : PISN data frame to add. If addPISN is false, you can ignore this argument
 
 ratioPISN : between 0 and 1, gives the number of PISN to add to a training sample OR to substract to a testing sample. 
 If ratioPISN = -1 then all PISN we be added to a training sample and no PISN will be substracted to a testing sample
 
 training : True for a training sample, False for a testing sample. It specifies the data set for the PISN to be added
 
 band : array like of all the passband you want to keep (ex/ [0,1,2,3,4,5] is to keep them all)
 
 name : name given to the saved .pkl file at the end
 
 dff : only deep drilling field ?
 
 extra : only extra galactic objects ?
 
 Dbool : only detected boolean ?
 
 complete : keep only objects that have a minimum of 'mini' points in EVERY chosen passband. 
 
 mini : minimum number of points in a passband (only the one chose in 'band') to be consider exploitable
 
 norm : normalise the 'mjd' column by translating it to zero ?

  
## Parametrise dataset

Once you have your dataset, the idea is to fit the lightcurves using a given model. The parameters used for the fit (for each passband of a given object) will be used for the machine learning step. For example from a simple polynomial fit of the form A*x^2 + B*x + C,  we will extract 3 parameters per passband per object.
So, from a data set we need to obtain a table with all the parameters for each objects. In the folder FeatureExtraction is a script "paraa" that allows. It contains a functions 'parametrise' that will return you a table of parameter with the associated objects.

  parametrise(train,nb_param,band_used,guess,err,save,checkpoint='',begin=0):
 
 

 train : lightcurves dataframe to parametrize
 
 nb_param : number of parameter in your model
 
 band_used : array of all band used (ex : [2,3,4])
 
 guess : array of all initial guess for the parameters : guess [1, 0, 1, 30, -5] is good for bazin
 
 err : the err function associated with your model
 
 checkpoint : the table is saved each time a ligne is calculated, if a problem occured you can put the partially filled table as a check point. With the right  'begin' it avoids recalculating from start.
 
 begin : first object to parametrise (in case previous parametrisation you had a problem)
 
 save : location and name of the save

Here we tried with two models, the polynomial previous mentionned and the Bazin function (more here : https://arxiv.org/pdf/0904.1066.pdf) 

## Machine learning

Once we have the parameters table, most of the work is done. We are using a random forest algorithm to train the model an observe the results. The analysis notebooks are in the folder MachineLearning
