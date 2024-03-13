# Start the Recommender System

1. create a virtual python environment on Python Version
   3.10 (https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)
2. Run *pip install -r requirements.txt* to get all the requiren libraries
3. May need to add the local python package to the current Python Path by typing: *export PYTHONPATH="${PYTHONPATH}:
   /Users/user1/Desktop/.../investment-locations-lstm"* to the terminal
4. Run the app.py file by typing: *streamlit run application/app.py*


## Get The Data
Most of the code is in Python and the data is stored in a .csv format. Due to file size not .csv files will be included
in this project. In the data folder you will get the RData files and some Excel sheets explaining the variables. Open the
.Rmd file in the root directory. With this you will get the csv files from the RData frame. Then follow the workflow.


## Preprocessing
Install jupyter notebook (https://jupyter.org/) on your device to run the code here. Run jupyter notebook on your device
and open the exploration.ipynb file. This is for exploring the data and find some first patterns. Then open the preprocess
notebook. Here you need to run every cell to get the processed data stored as .csv on your device. After this stage you can start
with the model training scripts.

## Train and evaluate the Models

For that you need to go into */application/main.py* In the main method you can load the dataset and start the model
training as well as the model evaluation. This may take a while.
The LSTM model is built with hyper paramter tuning, you can delete the */unlimited_project* folder to reset the model to
the dumy model. When running the model building stage now it will also test different parameters to find the best
fitting
model on a subset of the data.

## Depoloy the Recommender System
After teh model is build you can test the streamlit application to get your first recommendations. A dockerfile is also
included if you want to build an image and run the program inside a linux container. Run *streamlit run application/app.py*
in your root directory to start the user interface and test the model. All required packages need to be installed first.



## Evaluation
This is also a jupyter notebook you can use together with the recommender system. There we evaluate the recommendations
given by the models. You can find similar companies based on your chosen one and you can dive deeper in the behaviour of the models.
docker build -t my_streamlit_app .
docker run -p 8501:8501 my_streamlit_app
