clear all
close all
clc

videos_path='\Video_flash';
temp_path='\temp_imgs';
features_path='\video_features_4096';

%//////////////////////////////////////////////////////////////////////
faceDetector = vision.CascadeObjectDetector();
trainedNet=load('trained_vggface.mat','trainedNet');
trainedNet=trainedNet.trainedNet;
video_files=dir(videos_path);
for i=1:length(video_files)
    try
       curr_file=video_files(i).name;

       out_folder_path=fullfile(features_path,curr_file(1:end-4));
       out_path=[out_folder_path '.xlsx'];

        v_files=dir(temp_path);
        for j=3:length(v_files)
           delete(fullfile(temp_path,v_files(j).name));
        end

       sys_cmd=['ffmpeg -hwaccel cuda -i "' fullfile(videos_path,curr_file) '" -vf fps=30 -q:v 1 "' fullfile(temp_path,'exp_%03d.png') '"'];
       system(sys_cmd);

       %////////////////////////////////////////////////////////
        imageFiles = dir(fullfile(temp_path, '*.png'));  % Change to '*.jpg' if needed
        numImages = numel(imageFiles);

        % Initialize
        imagesCell = cell(1, numImages);
        faceDetector = vision.CascadeObjectDetector();
        for j = 1:numImages
            imgPath = fullfile(temp_path, imageFiles(j).name);
            img = imread(imgPath);

            % Face detection
            grayImg = rgb2gray(img);
            bboxes = faceDetector.step(grayImg);

            if ~isempty(bboxes)
                faceImg = imcrop(img, bboxes(1, :));
                faceImg = imresize(im2double(faceImg), [224 224]);
            else
                faceImg(1:224,1:224,1:3)=0;
                faceImg=uint8(faceImg);
            end
            imagesCell{j} = faceImg;
        end

        %////////////////////////////////////////////////////////
        featureLayer = 'fc7'; 
        allImages4D = cat(4, imagesCell{:});
        allImages4D = gpuArray(allImages4D);

        features = activations(trainedNet, allImages4D, featureLayer, 'ExecutionEnvironment', 'gpu');
        features = squeeze(mean(features, [1 2]));
        features = gather(features);

        xlswrite(out_path,double(features))
    catch
    end
end

%/////////////// reducing size of final features using Linearization /////
out_path='\video_features_128';
outfeatures_size=128;
files=dir(features_path);
parpool(10);
parfor i=3:length(files)
    try
        curr_file=fullfile(features_path,files(i).name);
        T=readtable(curr_file);
        T=table2array(T);
    
        infeatures_size = size(T,1); 
        segmentSize = infeatures_size / outfeatures_size;
        feature1=averaging(T,outfeatures_size,segmentSize);
        
        out_path_curr=fullfile(out_path,files(i).name);
        xlswrite(out_path_curr,double(feature1))
    catch

    end
end


function feature1=averaging(T,outfeatures_size,segmentSize)
    for k=1:size(T,2)
        for j = 1:outfeatures_size
            startIdx = round((j - 1) * segmentSize) + 1;
            endIdx = round(j * segmentSize);
            feature1(j,k) = mean(T(startIdx:endIdx,k));
        end
    end
end
