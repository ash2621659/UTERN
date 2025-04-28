clear all
close all
clc

videos_path='\Video_flash';
temp_path='\temp_imgs';
cords_temp='\cords_temp';
dummy_face_temp='\dummy_face_temp';
features_path='\video_features_2';

faceDetector = vision.CascadeObjectDetector();
areas_marking=face_points();
faceDetector = vision.CascadeObjectDetector();
areas_marking=face_points();
%//////////////////////////////////////////////////////////////////////

video_files=dir(videos_path);
for i=3:length(video_files)
    %try
       curr_file=video_files(i).name;

       out_folder_path=fullfile(features_path,curr_file(1:end-4));
       out_path=[out_folder_path '.xlsx'];

       v_files=dir(temp_path);
       for j=3:length(v_files)
           delete(fullfile(temp_path,v_files(j).name));
       end
       sys_cmd=['ffmpeg -hwaccel cuda -i "' fullfile(videos_path,curr_file) '" -vf fps=30 -q:v 1 "' fullfile(temp_path,'exp_%03d.png') '"'];
       system(sys_cmd);
      
       extracted_files=dir(temp_path);
       clear features
       tic
       parfor j=3:length(extracted_files)
           try
            curr_frame=imread(fullfile(temp_path,extracted_files(j).name));
            crop_face=curr_frame;
            out_weights=media_pipe_1(dummy_face_temp,cords_temp,crop_face,areas_marking);
            features(:,j-2)=out_weights;
           catch

           end
       end
       for ii=1:10
           try
                xlswrite(out_path,double(features))
                break
           catch
               pause(1)
           end
       end
    % catch
    % 
    % end
end