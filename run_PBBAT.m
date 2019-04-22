%   This function runs the SPBACF tracker on the video specified in "seq".
%   This function borrowed from BACF paper. 

function results = run_PBBAT(seq, video_path, lr)

% HOG feature parameters
hog_params.nDim   = 31;         
params.video_path = video_path; 
% Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;         

% Global feature parameters 
params.t_features = {
    ...struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
%     struct('getFeature',@get_fhog,'fparams',hog_params),...                 
};
params.t_global.cell_size = 4;                  % Feature cell size         
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases 

% Search region + extended background parameters
params.search_area_shape = 'square';            % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5;                   % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;                % the size of the training/detection area in feature grid cells

% Learning parameters
params.learning_rate       = lr;                % learning rate 
params.output_sigma_factor = 1/16;              % standard deviation of the desired correlation output (proportional to target)

% Detection parameters
params.interpolate_response  = 4;               % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations     = 20;              % number of Newton's iteration to maximize the detection scores 

% Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.01;

% Size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2); 
    
params.s_frames = seq.s_frames; 
params.no_fram  = seq.en_frame - seq.st_frame + 1; 
params.seq_st_frame = seq.st_frame;
params.seq_en_frame = seq.en_frame; 

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function. 
params.admm_iterations = 2; 
params.admm_lambda = 0.03; 

% Parameters for particle filter
params.numsample = 300; 
params.affsig = [4,4,0.04,0,0,0]; 

% Network setting
global enableGPU;
enableGPU = true;

params.indLayers = [19];   
params.nweights  = [1]; 
params.numLayers = length(params.indLayers);

vl_setupnn();
% vl_compilenn('enableGpu', true);

% Debug and visualization 
params.visualization = 1; 

% Run the main function
results = PBBAT_optimized(params);
