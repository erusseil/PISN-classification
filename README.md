# PISN-classification

Hello user !

The Large Synoptic Survey Telescope (LSST) will soon be operational and will provide us with an inimaginable amount of data. To be more specific, it will scan the sky, looking for objects with variable luminosity over time. Since analysing all the light curves manually is litteraly impossible, we need to figure a way to filter the data.

What if we want to predict the type of object that we are looking at only using it's light curve ? For this we need to train a machine learning algorithm with a lot of data. Fortunately a simulation of the data that LSST could produce in 3 years exists : it is called PLAsTiCC.

It was original created for a Kaggle challenge in 2018. More here details [here](https://www.kaggle.com/c/PLAsTiCC-2018/overview).

The training sample was composed of 7,846 objects considered to be spectroscopically confirmed and the testing sample was composed of about 3,492,888 objects, corresponding to a photometric-only sample. The difficulty comes from the statistical differences between these two data sets. Training and test samples are not representative of each other, thus breaking the initial assumptions from supervised machine learning. 

Moreover, the data is not only statistically different, training and testing samples also hold different number of classes (or populations). 
Since LSST will be more sensitive than any other telescope in history, it is reasonable to expect that it will detect objects that we've never observed before. In order to mimic this situation, a few very rare objects were added to the PLAsTiCC testing sample (only !). The organizers of PLAsTiCC hoped that participants would use some kind of anomaly detection techniques to mark as 'unknown' objects that were only in the testing sample. This task did not attract many participants. However, this is a very important task and if we fail to identify previously unknown objects, new physics could slip between our hands !

It is in this context that this project takes place. We will focus on identifying Pair Instability Supernovae (called PISN) in particular but in principle the method is wide enough to be applied to other objects. PISN are a rare type of transient event which was not present in the PLAsTiCC training sample. Now that the testing sample is public, we will use the PISN data to train directly a machine learning model. 

To begin with, you need to have the PLAsTiCC data set. It is downladable [here](https://zenodo.org/record/2539456#.YED0lP4o9hE)

For both testing and training you will find a data and a metadata file. The data file contains the light curves themselves, while metadata contains additionnal informations about the objects.
The training is made of 2 files : "plasticc_train_lightcurves.csv" and "plasticc_train_metadata.csv" 
The testing is made of 12 files : 11 "plasticc_test_lightcurves_xx.csv" and "plasticc_test_metadata.csv"

In addition I used the data of all PISN as separate files [here](https://drive.google.com/file/d/16_G2IjpJVdiv6GT0fs61-C_NuhHCPH8E/view)

In the folder FilterDataBase you can find a notebook 'FusePISN.ipynb' that fuses all PISN files into one dataframe.

Once all files are downloaded we are ready. We can distinguish four main steps : Split ; Filter ; Parametrise ; Predict

## Splitting the dataset

This first step in optional. Instead of using the dataset as it is one might want to divide each light curve into two smaller light curves. This results in more objects with less points per object. We then keep only objects that have at least 1 point with signal in one of it's passband. The splitting function is stored in the file splitting.py and can be executed through the file split_database.ipynb. The idea is to create the first models without splitting the data, and assuming it gives good results, then challenge it on a harder dataset.

## Filter dataset

There are many ways to improve your dataset by simply filtering 'useless' objects. In the folder FilterDataBase there is the script "data_base.py" that contains a functions 'create' that will return you a clean filtered dataset. At each step informations about the filters applied and remaining objects are printed. All those informations are saved in a txt file. This function gives control over a variety of option detailed below : 

### Data transformation

The first thing we usually do with light curves is to translate the time to 0. This means that we set the time of the first point to be 0, and we apply the same translation to all the points on the curve.
Additionaly we might want to normalize the flux by dividing the values by the maximum. 

### Data filtering
One might want to apply specific cuts, to filter certain objects. You can :

--> Keep only the passbands needed

--> Choose galactic or extragalactic objects

--> Choose deep drilling field or wide fast deep region

--> Keep only points marqued as detected boolean

--> Add PISNe to your dataset

Once every filter is applied, you get a final (also optional) step where we check object 'completeness'. It means that if a given object has less than a minimum of points (to be inputed) in each of the chosen passband, then it is removed from the final dataset.


  
## Parametrise dataset

Once you have your dataset, the idea is to fit the lightcurves using a given model. The parameters used for the fit (for each passband of a given object) will be used for the machine learning step. For example from a simple polynomial fit of the form A*x^2 + B*x + C,  we will extract 3 parameters per passband per object. Additionally we have the option of adding te following extra parameters: the number of points; the maximum peak value; the loss value for the fit; and the number of detected boolean points. Assuming we keep all the passbands and all the extra parameters, in this example we would get 6 * 7 = 42 parameters for each object.

In the end, from a dataset we need to obtain a rectangular table with all the parameters for each objects. In the folder FeatureExtraction is a script "parametrisation.py" that allows that. It contains a functions 'parametrise' that will return you a table of parameter with the associated objects. Along this file you can also find "models.py" that contains the mathematical models for the fit : you can try other fits by adding your functions in this file.
Here we tried with two models, the polynomial previously mentionned and the Bazin function (more [here](https://arxiv.org/pdf/0904.1066.pdf)) 

## Machine learning prediction

Once we have the parameters table, most of the work is done. Two kind of analysis may come from this : 

### Classification

This is traditional classification machine learning using a Random Forest. In order to do this we are using the random forest algorithm from sklearn. In the same file PISNml.py, there is a function create_ml that will train your model and saves it. You can then visualise the results of your model in the file ModelScore.ipynb. Be aware that supervised classification is not really suited to detect few objects hidden in a large dataset, the results we obtained for PISNe are not very conclusive. Anomaly detection should be prefered in this case.

### Anomaly detection

Since there are so few PISN,  anomaly detection algorithm is suited for the job. The idea is that PISN might be so different from other objects that they are isolated in the space parameter. We could then focus only on the most isolated objects and hope to find PISN.

To accomplish this we are using the isolation forest algorithm from sklearn. In the folder MachineLearning you can find a script "PISNml.py" that contains a function create_if. Inputing the parameter table in this function will incremente the table with extra columns caracterising how normal the object is comparing to other objects. The lower this score is, the more isolated is the object is in the parameter space. The creation and visualisation of your results are available in file Isolation forest.ipynb. Additionnaly you can visualise the space parameter in the jupyter notebook ParamStudy.ipynb. This method provides good results to identify high signal PISNe.


## Discussion

The pipeline is operational and experiments can be done using different fits, different filters and different parametrisations.
This work took place in the context of a master 2 internship and thus, all the results I obtained are detailed in the file PISN_report.pdf.
Additionnal results were produced after the report : the main contaminant of the isolation forest being Active Galactic Nuclei (AGNs), we tried to apply the random forest classification to detect them. It turns out that the model we used (quadratic fit, 4 extra parameters and WFD sample) is extremely efficient at identifying AGNs. We can therefore use this model to filter the AGNs and then upgrade the results for the detection of PISNe.
