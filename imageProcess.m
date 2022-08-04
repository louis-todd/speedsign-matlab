trainDatasetPath = fullfile(pwd, 'Train'); %getting file path for training data
[trainMatrix, trainLabelVector]  = prepare(trainDatasetPath); %parsing training file path into prepare function
testDatasetPath = fullfile(pwd, 'Test'); %getting file path for testing data
[testMatrix, testLabelVector] = prepare(testDatasetPath); %parsing testing file path into prepare function

%calling classification algorithm and parsing the training feature matrix
%and label vector, to get model back. Using tic toc to measure time it takes:
tic;
model = fitcdiscr(trainMatrix,trainLabelVector); 
t = toc;
%getting predicitons from parsing in model and test feature matrix:
predictions = predict(model,testMatrix);

%creating confusion matrix, with test tabel vector and predicitons.
cm = confusionchart(testLabelVector, predictions);
cm.RowSummary = 'row-normalized';

%count how many predictions were right
count = 0;
for i = 1:69
    if predictions(i) == testLabelVector(i)
        count = count + 1;
    end
end
%calculate accuracy as percentage:
accuracy = count/69*100;
% function that prepares the features matrix and label vector, with the
% given dataset path:
function [Matrix, Vector] = prepare(DatasetPath) 
    % creates image data store, with labels from parent folder name:
    imds = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imNo = numel(imds.Files); % gives the number of images in the imageDataStore
    Matrix = zeros(imNo, 4096); % creating blank matrix
    Vector = categorical.empty(imNo,0); % creating blank vector

    for i = 1:imNo % iterates through images in imageDataStore
        current = imds.Files{i}; % current image
        im = mean(double(imread(current))/255,3); % import image with type double, in grayscale
        im = reshape(im,1,[]); % convert to row vector
        Matrix(i, :) = im; % inserts prepared images into matrix
        Vector(i) = imds.Labels(i); % inserts label into label vector
    end
end