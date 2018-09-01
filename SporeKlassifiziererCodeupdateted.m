%% Clean up the workspace
% We use this three functions to start with a fresh and clean workspace
clear 
clc 
close all 
%% Load alex net 
% We call the already pre trained network 'alex net'
alex = alexnet;
% Define a variable called layers to store alex nets layers
layers = alex.Layers;
% To use alex nets abilities we need to modify the 23th and 25th layer 
layers(23) = fullyConnectedLayer(3);
layers(25) = classificationLayer ;
%% Define a Path
addpath(genpath('C:\Users\Jeri\Documents\MATLAB\Sporetestobjekte'));
%% Working with hyperparameters
% We want to secure that the training is done using the gpu
hyperparam.ExecutionEnvironment = 'gpu';
%% Modify pictures 
% Alex net can only read images which are saved in the apropriate format 
% To resize the images to 227*227*3 we prepare a variable called
% 'imagesize' furthermore we define an augmenter which we will use on our
% imageDatastore 
augmenter = imageDataAugmenter('RandXReflection',true,'RandXScale',[1 2],'RandYReflection',true,'RandYScale',[1 2]);
imageSize = [227 227] ;
%% Create imagedatastore to store the pictures of our spore creatures 
% We define our imagedatastore 'Images'
% Our imageDatastore needs to include the Subfolders and to extract the
% names of our subfolders to sort the images according to the names of our
% subfolders
allImages = imageDatastore('C:\Users\Jeri\Documents\MATLAB\Sporetestobjekte', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Split the images into three categories 
% Pass the Labels (which we extracted earlier) towards the different
% pictures 
[trainingImages,validationImages, testImages] = splitEachLabel(allImages, 0.7, 0.15, 0.15 );
numberofTrainingPictures = countEachLabel(trainingImages);
numberofTestPictures = countEachLabel(testImages);
numberofValidationPictures = countEachLabel(validationImages);
testPictureCategories = numberofTestPictures.Label;
trainingPictureCategories = numberofTrainingPictures.Label;
validationPictureCategories = numberofValidationPictures.Label;
%% Apply the Imageaugmenter 
% We create a augmentedImageDatastore for our trainingImages
% We apply our predifened imageSize, since alexnet requiers 227*227 pictures
au_re_TrainingImages = augmentedImageDatastore(imageSize,trainingImages,'DataAugmentation',augmenter);
re_TestImages = augmentedImageDatastore(imageSize, testImages);
re_ValidationImages = augmentedImageDatastore (imageSize, validationImages);
%% Defining the training options 
% Epoch = Repeats of the whole training sets
% Minibatch = Repeats of a part of the training set
opts = trainingOptions('sgdm', 'Initiallearnrate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64, 'ValidationData',re_ValidationImages, 'plots', 'training-progress');
t = tic; % starts a timer 
myNet = trainNetwork(au_re_TrainingImages, layers, opts);
usedTimeForTraining = toc(t); % ends a timer we save the elapsed time in a variable
%% Test the trained net via the test images 
%Classify the test images (ensure that this also will run on the gpu)
predictedLabels = classify(myNet,re_TestImages,'ExecutionEnvironment',hyperparam.ExecutionEnvironment);
accuracy = mean(predictedLabels == testImages.Labels);
disp(accuracy)



