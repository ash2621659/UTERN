clear all
close all
warning off
clc

video_features_path1='\video_features_9_new';
video_features_path2='\video_features_128';
audio_features_path='\audio_features_13_new';
label_path='\labels_consensus_6class_D\label_consensus_1.csv';
results='\results_cremad';


%///// preparing dataset for training ////////////////////////////

T=readtable(label_path);
sample_names=table2array(T(:,1));
sample_labels=table2array(T(:,2:7));

for i=1:length(sample_names)
    curr_name=sample_names{i};
    k2=strfind(curr_name,'.');
    sample_names{i}=curr_name(1:k2(end)-1);

    curr_labels=sample_labels(i,:);
    [r,c]=find(curr_labels==max(curr_labels));
    category(i,1)=c(1);
end

v_files=dir(video_features_path1);

%/////////////////////////////////////////////////////////////////
min_frames=60;


parfor i=1:length(sample_names)
   % try
        disp(['importing sample no: ' num2str(i) '/' num2str(length(sample_names))])
        curr_name=sample_names{i};
        v_file_path1=fullfile(video_features_path1,[sample_names{i} '.xlsx']);
        v_file_path2=fullfile(video_features_path2,[sample_names{i} '.xlsx']);
        a_file_path=fullfile(audio_features_path,[sample_names{i} '.xlsx']);

        if (exist(v_file_path1, 'file') && exist(v_file_path2, 'file') && exist(a_file_path, 'file'))
           category_o(i,1)=category(i,1);

           v_features1=readtable(v_file_path1);
           v_features1=table2array(v_features1);
           if(size(v_features1,2)<min_frames)
               v_features1(:,end:min_frames)=0;
           else
               v_features1=v_features1(:,1:min_frames);
           end

           v_features2=readtable(v_file_path2);
           v_features2=table2array(v_features2);
           if(size(v_features2,2)<min_frames)
               v_features2(:,end:min_frames)=0;
           else
               v_features2=v_features2(:,1:min_frames);
           end

           a_features=readtable(a_file_path);
           a_features=table2array(a_features);
           if(size(a_features,2)<min_frames)
               a_features(:,end:min_frames)=0;
           else
               a_features=a_features(:,1:min_frames);
           end

           %//////// features concatenation ////////////////
            X=[v_features1;v_features2];
            Y=a_features;
            [lagsAll,correlation]=CCA(X,Y);
            lagsAll=abs(lagsAll);
            if(lagsAll>0)
                a_features=[a_features(:,lagsAll+1:end) repmat(0,size(a_features,1),lagsAll)];
                av_features=[v_features1;v_features2;a_features];
            else
                av_features=[v_features1;v_features2;a_features];
            end
            cca_mat(i,:)=[correlation,lagsAll];

            v_features=[v_features1;v_features2];



          audiovideoFeaturesAll(:,:,1,i)=av_features; 
          videoFeaturesAll(:,:,1,i)=v_features; 
          audioFeaturesAll(:,:,1,i)=a_features; 
        end
    % catch
    % 
    % end
end

videoFeaturesAll=videoFeaturesAll*100;
emotionLabels = categorical(category_o);

[r,c]=find(category_o==0);
emotionLabels(r)=[];
audiovideoFeaturesAll(:,:,:,r)=[];
videoFeaturesAll(:,:,:,r)=[];
audioFeaturesAll(:,:,:,r)=[];
cca_mat(r,:)=[];

lag_avg=mean(max(abs(cca_mat(:,2))));
corr_avg=mean(abs(cca_mat(:,1)));
disp(['Average lag using CCA :' num2str(lag_avg * 30)])
disp(['Agerage correlation :' num2str(corr_avg)])

%////////// balancing dataset ///////////////

[C, ~, ic] = unique(emotionLabels);
counts = accumarray(ic, 1);
minCount = min(counts);


% Initialize
audiovideoFeaturesBalanced = [];
videoFeaturesBalanced = [];
audioFeaturesBalanced = [];
emotionLabelsBalanced = [];
for i = 1:numel(C)
    idx = find(emotionLabels == C(i)); % Indices for this class
    idx = idx(randperm(numel(idx))); % Shuffle indices
    idx = idx(1:minCount); % Pick minCount samples

    % Select and concatenate along the 4th dimension
    audiovideoFeaturesBalanced = cat(4, audiovideoFeaturesBalanced, audiovideoFeaturesAll(:,:,:,idx));
    videoFeaturesBalanced = cat(4, videoFeaturesBalanced, videoFeaturesAll(:,:,:,idx));
    audioFeaturesBalanced = cat(4, audioFeaturesBalanced, audioFeaturesAll(:,:,:,idx));
    emotionLabelsBalanced = [emotionLabelsBalanced; emotionLabels(idx)];
end

% Optional: Shuffle the final balanced dataset
numSamplesBalanced = size(audiovideoFeaturesBalanced, 4);
shuffleIdx = randperm(numSamplesBalanced);

audiovideoFeaturesBalanced = audiovideoFeaturesBalanced(:,:,:,shuffleIdx);
videoFeaturesBalanced = videoFeaturesBalanced(:,:,:,shuffleIdx);
audioFeaturesBalanced = audioFeaturesBalanced(:,:,:,shuffleIdx);
emotionLabelsBalanced = emotionLabelsBalanced(shuffleIdx);


numVideos=size(audiovideoFeaturesBalanced,4);
idx2 = randperm(numVideos);
numTrain = round(0.8 * numVideos);

XTrain1 = audiovideoFeaturesBalanced(:, :, :, idx2(1:numTrain));
XTrain2 = videoFeaturesBalanced(:, :, :, idx2(1:numTrain));
XTrain3 = audioFeaturesBalanced(:, :, :, idx2(1:numTrain));
YTrain = emotionLabelsBalanced(idx2(1:numTrain));

XVal1 = audiovideoFeaturesBalanced(:, :, :, idx2(numTrain+1:end));
XVal2 = videoFeaturesBalanced(:, :, :, idx2(numTrain+1:end));
XVal3 = audioFeaturesBalanced(:, :, :, idx2(numTrain+1:end));
YVal = emotionLabelsBalanced(idx2(numTrain+1:end));


%///////////// audio video trainning////////////////////

numFrames=min_frames;
numFeatures=size(XTrain1,1);

layers = [
    imageInputLayer([numFeatures numFrames 1], 'Name', 'input')


    convolution2dLayer([5 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')

    convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')

    dropoutLayer(0.5, 'Name', 'dropout1')

    convolution2dLayer([3 1], 512, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(7, 'Name', 'fc1')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...      % reduce LR by 50%
    'LearnRateDropPeriod', 10, ...       % every 10 epochs
    'MaxEpochs',60, ...
    'MiniBatchSize',64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal1, YVal}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


net = trainNetwork(XTrain1,YTrain, layers, options);
predictedScores= predict(net,XVal1);
yTrue= double(YVal);
[~, maxIdx] = max(predictedScores, [], 2);
yPred = categorical(maxIdx, 1:numel(categories(emotionLabels)), categories(emotionLabels));
yPred=double(yPred);
com1=[yTrue yPred];

xlswrite(fullfile(results,'run1_audiovisual.xlsx'),com1);
disp('Audio video')
[F1_per_class, F1_macro, F1_micro] = computeF1ScoreMicroMacro(yTrue, yPred);


%///////////// video trainning////////////////////

numFrames=min_frames;
numFeatures=size(XTrain2,1);

layers = [
    imageInputLayer([numFeatures numFrames 1], 'Name', 'input')


    convolution2dLayer([5 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')

    convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')

    dropoutLayer(0.5, 'Name', 'dropout1')

    convolution2dLayer([3 1], 512, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(7, 'Name', 'fc1')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...      % reduce LR by 50%
    'LearnRateDropPeriod', 10, ...       % every 10 epochs
    'MaxEpochs',12, ...
    'MiniBatchSize',64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal2, YVal}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


net = trainNetwork(XTrain2,YTrain, layers, options);
predictedScores= predict(net,XVal2);
yTrue= double(YVal);
[~, maxIdx] = max(predictedScores, [], 2);
yPred = categorical(maxIdx, 1:numel(categories(emotionLabels)), categories(emotionLabels));
yPred=double(yPred);
com2=[yTrue yPred];

xlswrite(fullfile(results,'run1_visual.xlsx'),com2);
disp('Video only')
[F2_per_class, F2_macro, F2_micro] = computeF1ScoreMicroMacro(yTrue, yPred);


%///////////// audio trainning////////////////////

numFrames=min_frames;
numFeatures=size(XTrain3,1);

layers = [
    imageInputLayer([numFeatures numFrames 1], 'Name', 'input')


    convolution2dLayer([5 1], 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([3 1], 128, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')

    convolution2dLayer([3 1], 256, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')

    dropoutLayer(0.5, 'Name', 'dropout1')

    convolution2dLayer([3 1], 512, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    globalAveragePooling2dLayer('Name', 'gap')

    fullyConnectedLayer(7, 'Name', 'fc1')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...      % reduce LR by 50%
    'LearnRateDropPeriod', 10, ...       % every 10 epochs
    'MaxEpochs',15, ...
    'MiniBatchSize',64, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal3, YVal}, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


net = trainNetwork(XTrain3,YTrain, layers, options);
predictedScores= predict(net,XVal3);
yTrue= double(YVal);
[~, maxIdx] = max(predictedScores, [], 2);
yPred = categorical(maxIdx, 1:numel(categories(emotionLabels)), categories(emotionLabels));
yPred=double(yPred);
com3=[yTrue yPred];

xlswrite(fullfile(results,'run1_audio.xlsx'),com3);
disp('Audio only')
[F3_per_class, F3_macro, F3_micro] = computeF1ScoreMicroMacro(yTrue, yPred);




%///////// computing F1 /////////////////////////////////////////////
function [F1_per_class, F1_macro, F1_micro] = computeF1ScoreMicroMacro(yTrue, yPred)
% Computes per-class F1, macro-averaged F1, and micro-averaged F1
% Inputs:
%   yTrue - true labels (categorical or cell array)
%   yPred - predicted labels (categorical or cell array)

    % Convert to categorical if needed
    if ~iscategorical(yTrue)
        yTrue = categorical(yTrue);
    end
    if ~iscategorical(yPred)
        yPred = categorical(yPred);
    end

    % Ensure both have the same categories
    allClasses = union(categories(yTrue), categories(yPred));
    yTrue = categorical(yTrue, allClasses);
    yPred = categorical(yPred, allClasses);

    % Initialize
    numClasses = numel(allClasses);
    F1_per_class = zeros(numClasses, 1);

    % For micro score: accumulate TP, FP, FN
    TP_micro = 0;
    FP_micro = 0;
    FN_micro = 0;

    % Loop through each class
    for i = 1:numClasses
        classLabel = allClasses{i};

        % One-vs-rest calculation
        TP = sum((yTrue == classLabel) & (yPred == classLabel));
        FP = sum((yTrue ~= classLabel) & (yPred == classLabel));
        FN = sum((yTrue == classLabel) & (yPred ~= classLabel));

        % Precision and Recall
        precision = TP / (TP + FP + eps);
        recall    = TP / (TP + FN + eps);

        % F1 for this class
        F1_per_class(i) = 2 * (precision * recall) / (precision + recall + eps);

        % Accumulate for micro
        TP_micro = TP_micro + TP;
        FP_micro = FP_micro + FP;
        FN_micro = FN_micro + FN;
    end

    % Macro-average: mean of class-wise F1s
    F1_macro = mean(F1_per_class);

    % Micro-average: global precision and recall
    precision_micro = TP_micro / (TP_micro + FP_micro + eps);
    recall_micro    = TP_micro / (TP_micro + FN_micro + eps);
    F1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + eps);
end





