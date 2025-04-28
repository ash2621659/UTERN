clear all
close all
clc

videos_path='\CREMA_D';
temp_audio_path='\temp_audio';
features_path='\audio_features_2';

video_fps=30;
video_files=dir(videos_path);
for i=3:length(video_files)
   curr_file=video_files(i).name;
   in_audio_path=fullfile(videos_path,curr_file);
   out_audio_path = fullfile(temp_audio_path, 'converted.mp4');

   a_files=dir(temp_audio_path);
   for j=3:length(a_files)
       delete(fullfile(temp_audio_path,a_files(j).name));
   end

   sys_cmd = ['ffmpeg -hwaccel cuda -i "' in_audio_path '" -q:a 0 -map a "' out_audio_path '"'];
   system(sys_cmd)

   [audioData, fs] = audioread(out_audio_path);
   out_frames=(length(audioData)/fs) * video_fps;
   mfccPerFrame=extractMFCC120Frames(audioData, fs,out_frames);

   out_folder_path=fullfile(features_path,curr_file(1:end-4));
   out_path=[out_folder_path '.xlsx'];
   xlswrite(out_path,mfccPerFrame')
end




function mfccPerFrame = extractMFCC120Frames(audioData, fs, numFrames)
    
    % Normalize
    audioData = audioData / max(abs(audioData));

    % Extract MFCCs
    coeffs = mfcc(audioData, fs, ...
        'NumCoeffs', 13, ...
        'LogEnergy', 'Ignore', ...
        'WindowLength', round(0.025 * fs), ...
        'OverlapLength', round(0.015 * fs));  % typical 25ms window, 10ms hop

    % Resize MFCCs to match number of frames (e.g., 120)
    mfccPerFrame = imresize(coeffs, [numFrames 13]);
end
