clear all
close all
clc


% Step 1: Load data
data1 = readmatrix('\audiovisual_final.xlsx');
data2 = readmatrix('\visual_final.xlsx');
data3 = readmatrix('\audio_final.xlsx');

actual = data1(:,1); % Actual class labels
pred1 = data1(:,2);  % Predicted class labels from model 1
pred2 = data2(:,2);  % Predicted class labels from model 2
pred3 = data3(:,2);  % Predicted class labels from model 3

numSamples = length(actual);
numClasses = max(actual); % Assuming classes are 1,2,...,numClasses

% -------------- Step 2: Convert to One-Hot Encoding --------------
onehot1 = zeros(numSamples, numClasses);
onehot2 = zeros(numSamples, numClasses);
onehot3 = zeros(numSamples, numClasses);

for i = 1:numSamples
    onehot1(i, pred1(i)) = 1;
    onehot2(i, pred2(i)) = 1;
    onehot3(i, pred3(i)) = 1;
end

% -------------- Step 3: Define Fitness Function --------------
fitnessFunction = @(w) classificationError(w, actual, onehot1, onehot2, onehot3);

% -------------- Step 4: GA Settings --------------
nvars = 3; % Three weights
lb = [0 0 0]; % Lower bounds
ub = [1 1 1]; % Upper bounds
Aeq = [1 1 1]; % Sum of weights = 1
beq = 1;

options = optimoptions('ga','PopulationSize', 10,'MaxGenerations', 50); 

% -------------- Step 5: Run GA Optimization --------------
[bestWeights, bestError] = ga(fitnessFunction, nvars, [], [], Aeq, beq, lb, ub, [], options);
load('wt')
% -------------- Step 6: Final Prediction and Accuracy --------------
combinedScores = bestWeights(1)*onehot1 + bestWeights(2)*onehot2 + bestWeights(3)*onehot3;
[~, finalPredictedClass] = max(combinedScores, [], 2);

accuracy = sum(finalPredictedClass == actual) / numSamples;

fprintf('Optimized Weights: %.4f %.4f %.4f\n', bestWeights);
fprintf('Classification Accuracy: %.2f%%\n', accuracy*100);

[F1_per_class, F1_macro, F1_micro] = computeF1ScoreMicroMacro(actual, finalPredictedClass)

% -------------- Step 7: Fitness Function Definition --------------
function error = classificationError(w, actual, onehot1, onehot2, onehot3)
    combinedScores = w(1)*onehot1 + w(2)*onehot2 + w(3)*onehot3;
    [~, predictedClass] = max(combinedScores, [], 2);
    error = 1 - (sum(predictedClass == actual) / length(actual)); % 1 - Accuracy
end



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





