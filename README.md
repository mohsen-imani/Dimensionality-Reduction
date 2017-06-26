# Dimensionality-Reduction

This project is able to recognize a person's face by comparing facial images to that of a known person.  The algorithm projects the image onto a "face space" composed of a complete basis of "eigenfaces."

Dataset:
 http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

There 40 subjects in this datasets and each subject has ten images. The size of image is 112 x 92 pixels.

Tasks:

1. Classification using KNN (1NN in this project) with the following techniques:
1.1 Using k-fold cross validation
1.2 In each cross validation, using PCA to reduce the dimensionality of images
3. Apply LDA to replace PCA for dimensionality reduction and repeat Task 1.

4. Run PCA first to reduce the image dimensionality to the number of training data, then using LDA to reduce the image dimensionality.






