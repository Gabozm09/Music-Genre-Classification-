# Music Genre Classification

This project is a simple music genre classification using
the GTZAN dataset and a Support Vector Machine (SVM) 
classifier.

## Requirements
* Python 3.7 or later
* Libraries
    * numpy
    * pandas
    * scikit-learn 
    * librosa 
    * multiprocessing

You can install the required libraries using the following command:
```bash
pip install numpy pandas scikit-learn librosa
```

### Dependencies
* resampy
* libsndfile
* libvorbis

```bash
pip install resampy libsndfile libvorbis
```

## Dataset

The GTZAN dataset used for this project can be downloaded from 
this link. After downloading and extracting the dataset, place 
the genres_original folder inside the Data folder in the project 
directory.

## Running the Code
Songs need to be on a wav format.

ex: song.wav

To run the code, simply execute the Music_test.py script:

```bash
python Music_test.py
```
The script will load the GTZAN dataset, preprocess the data,
train an SVM classifier, and evaluate the model's performance
using a confusion matrix and a classification report. It will
also predict the genre of a sample song.

You can change the new_song variable in the script to test 
the model on different songs.
