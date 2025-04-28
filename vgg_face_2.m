clear all
close all
clc

% Path to your dataset
imageFolder = '\SortedImages';

% Create image datastore
imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x)imresize(im2double(imread(x)), [224 224]));

% Split data: 80% training, 20% validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');


numClasses = numel(categories(imdsTrain.Labels));
layers = [
    imageInputLayer([224 224 3], 'Name', 'input')

    % Block 1
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1_1')
    reluLayer('Name', 'relu1_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    % Block 2
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv2_1')
    reluLayer('Name', 'relu2_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    % Block 3
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv3_1')
    reluLayer('Name', 'relu3_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    % Block 4
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv4_1')
    reluLayer('Name', 'relu4_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

    % Block 5
    convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv5_1')
    reluLayer('Name', 'relu5_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool5')

    % Compression block (reduces channels from 512 → 50)
    convolution2dLayer(1, 50, 'Name', 'conv_reduce')    % 1x1 conv
    reluLayer('Name', 'relu_reduce')

    % Global average pooling (spatial → vector)
    globalAveragePooling2dLayer('Name', 'gap')

    % Final classification layer
    fullyConnectedLayer(numClasses, 'Name', 'fc_final', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];


% Set training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...      % reduce LR by 50%
    'LearnRateDropPeriod', 30, ...       % every 10 epochs
    'MaxEpochs', 250, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
trainedNet = trainNetwork(imdsTrain, layers, options);
save('trained_vggface_50','trainedNet')

%///////////////////////////////////////////////////////

% trainedNet_1=load('trained_vggface.mat','trainedNet');
% 
% img = imread('\SortedImages\happy\ffhq_3072.png');
% img = imresize(im2double(img), [224 224]);
% 
% % Extract features
% featureLayer = 'fc7';
% singleFeature = activations(trainedNet, img, featureLayer, 'OutputAs', 'rows');
