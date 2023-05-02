# 16664-Final-Project

To execute these files, download them and make sure that they are in the same directory with *test* and *trainval* folders.

The code works as follows:

1. Run *build_dictionary.py* to create a SIFT descriptor dictionary.
   This randomly selects 10% of the training samples to run SIFT and
   produces a *dictionary.npz* file.
2. Run *sift_predict.py* to predict labels for all images in the 
   *test* folder. This creates the descriptor of the test image using
   the same SIFT detector, and then the features will be compared to
   entries in the dictionary. The prediction will be made based on
   the 4 closest descriptors in the dictionary, and the majority gets
   to decide the final label.

Note that there are parameters for SIFT feature detector. Feel free to change it and generate better predictions.
   
