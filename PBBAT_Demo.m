%   This script runs the original implementation of Background Aware Correlation Filters (BACF) for visual tracking.
%   the code is tested for Mac, Windows and Linux- you may need to compile
%   some of the mex files.

%   Some functions are borrowed from other papers (SRDCF, CCOT, KCF etc)- and

%   This function runs the PBBAT tracker on the video specified in "seq".
%   This function is based on the BACF tracker. 
%   Modified by Fuling Lin (fuling.lin@outlook.com)
%   Modified by Yinqiang Zhang (yinqiang.zhang@tum.de)
 
function PBBAT_Demo(~)
    close all;
    clear;
    clc;
    
    addpath('./utils');
    addpath('./model');
    addpath('./implementation');
    addpath('./evaluation');
    addpath('./external/matconvnet/matlab');
    addpath('./external/mtimesx');
    addpath('./external/mexResize');
    addpath('./external/interp2');
    addpath('./external/imResampleMex');
    addpath('./external/gradientMex');
% -------------UAV123-----------------
    video_path_UAV123 = '.\UAV123_10fps\data_seq\';
    ground_truth_path_UAV123 = '.\UAV123_10fps\anno\';
    
    seqs=configSeqs(video_path_UAV123);
    for tt = 1:numel(seqs)
        video_name = seqs{tt}.name;
        seq = load_video_info_UAV123(video_name, video_path_UAV123, ground_truth_path_UAV123);
        video_path = seq.video_path;
        ground_truth = seq.ground_truth;
        
        %   Run PBBAT - main function
        learning_rate = 0.013;  %   you can use different learning rate for different benchmarks.
        results       = run_PBBAT(seq, video_path, learning_rate);
      end
end
