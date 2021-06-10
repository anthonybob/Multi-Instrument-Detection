# Multi-Instrument-Detection
Neural Network that detects multiple instruments in polyphonic audio

### Contributors: Grant Baker, Anthony Bobrovnikov

### Description of Project
For our project in this course we experimented with different architectures for instrument classification in polyphonic audio. We built a feedforward neural network, a convolutional neural network, and an attention based convolutional recurrent neural network. We used Medley-solos-DB (https://zenodo.org/record/3464194#.YME4jzplA5l) for classification of instruments where only one instrument was being played. We used mtg-jamendo-dataset (https://github.com/MTG/mtg-jamendo-dataset) for classification of instruments where multiple instruments were played at the same time. We used the feedforward neural network on the Medley-solos-DB dataset, and a convolutional neural network on the mtg-jamendo-dataset. 

For the Medley-solos-DB dataset, the data is given as wav files. The Spectogram.py file takes the data in from wav files, and converts each wav file into a spectogram. In addition, it downsamples the spectogram to a 150 x 150 image. The wav_file_test in Spectogram.py takes a wav_file converts it to a spectogram and displays the spectogram. Then it downsamples the spectogram, saves the spectogram to a file, loads the spectogram from the file and displays the downsampled spectogram loaded from the file. This tests both the quality of the downsampled spectogram, and whether we can read and write the spectogram from the file correctly.

For the mtg-jamendo-dataset, the dataset comes with mel spectrograms. The npy_file_test ensures we can read in the spectogram, downsample the spectogram, and write it to the file by displaying the spectogram before and after loading the downsampled spectogram from a file. All work on the mtg-jamendo-dataset can be found in the notebook. Originally, I downloaded the 229GB dataset of mel spectrograms locally on an external harddrive. Eventually, I purchased Google Colab Pro and storage. To transfer the dataset, I compressed the splits (train, validation, test) of the sub-datset including the instrument labelled tracks. This script can be found in zipdataset.py. You can run this script like so:
> python zipdataset.py split path_to_direcory

Model.py contains the implementation of the feedforward neural network on the Medley-solos-DB dataset. We use 3 hidden layers. Originally, we had intended to do a classification of 8 different instruments. However, since the training data was too small for most of the classes we only achieved around 63% test accuracy for this. We limited the training to classify between two classes (Piano and Violin) since these had significantly more training data. When this was done, we achieved a 98% accuracy on the test set. 

The jupyter notebook contains the implementation of the convolutional recurrent neural network. Each track was quite long, so I started working on an attention based network. The model makes predictions for 3 second snippets and aggregates the logits and loss for the whole track. This ensures we don't miss out on any of the instruments included in the track, since there are multiple labels per track. This model is not fully functional, but it just needs a proper data generator to complete the training loop. This type of dataset and this type of network has great potential, so for future work we would like to get this model to train properly.

## Example prediction:
In the root directory of our repository, there is a file called 'model' that is a saved model from training the feedforward network. If you run Model.py and give it a wav_file as a command line argument, it was classify the wav_file as either being a piano or a violin. 
> python Model.py wav_file

Our LICENSE file is in the root directory

