% This function implements the PBCAT tracker.

function [results] = PBBAT_optimized(params)
% Setting parameters for local use.
search_area_scale   = params.search_area_scale; 
output_sigma_factor = params.output_sigma_factor; 
learning_rate       = params.learning_rate; 
filter_max_area     = params.filter_max_area; 
nScales             = params.number_of_scales; 
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response; 
n_sample = params.numsample;
affsig = params.affsig;

features    = params.t_features;
video_path  = params.video_path; 
s_frames    = params.s_frames; 
pos         = floor(params.init_pos); 
target_sz   = floor(params.wsize);

visualization  = params.visualization; 
num_frames     = params.no_fram; 
init_target_sz = target_sz; 

sub_target_sz = floor(target_sz/2);
% The number of subregions
subw_num = 5; 
sub_init_pos = zeros(subw_num,2);
gamma = 1.0e-4;

% Set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size; 
search_area = prod(init_target_sz/ featureRatio * search_area_scale); 

% parameters for CNN
indLayers = params.indLayers;   
nweights  = params.nweights; 
numLayers = params.numLayers;

% When the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh') 
    if search_area < params.t_global.cell_selection_thresh * filter_max_area 
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area 
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% Target size at the initial scale
base_target_sz = target_sz / currentScaleFactor; 

% Window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor(base_target_sz * search_area_scale);                    % proportional area, same aspect ratio as the target
    case 'square' 
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2);  % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

% Set the size to exactly match the cell size
sz = round(sz / (featureRatio)) *featureRatio;
use_sz = floor(sz/featureRatio);

% Construct the label function- correlation output, 2D gaussian function,
% With a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/2/(featureRatio)))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% Construct cosine window 
cos_window = single(hann(floor(use_sz(1)))*hann(floor(use_sz(2)))');

% Calculate feature dimension 
try 
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try 
        im = imread(s_frames{1});
    catch
        %disp([video_path '/' s_frames{1}])
        im = imread([video_path '/' s_frames{1}]);
    end
    
end

if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% Compute feature dimensionality 
feature_dim = 0;
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor') 
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray') 
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
        feature_dim = feature_dim + features{n}.fparams.nDim;
    end
end

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)); 
    scaleFactors = scale_step .^ scale_exp; 
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

rect_position = zeros(num_frames, 4);
time = 0;

% Allocate memory for multi-scale tracking 
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8'); 
layer_responsef_padded= zeros(use_sz(1), use_sz(2), numLayers);
response = zeros(5, use_sz(1), use_sz(2), nScales);
responsef_padded= zeros(use_sz(1), use_sz(2), nScales);
small_filter_sz = floor(base_target_sz/featureRatio); 

loop_frame = 1;
for frame = 1:numel(s_frames)
    % Load image
    try
        im = imread([video_path '/img/' s_frames{frame}]);
    catch
        try
            im = imread([s_frames{frame}]);
        catch
            im = imread([video_path '/' s_frames{frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    tic(); 
    %do not estimate translation and scaling on the first frame, since we
    %just want to initialize the tracker there
    if frame > 1
        for nn = 1:subw_num
            for scale_ind = 1:nScales
                multires_pixel_template(:,:,:,scale_ind) = ...
                    get_pixels(im, sub_init_pos(nn,:), round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
                for ii = 1:numLayers
                    xtf = fft2(bsxfun(@times,get_features_CNN(multires_pixel_template(:,:,:,scale_ind),use_sz,indLayers(ii)),cos_window));
                    responsef = permute(sum(bsxfun(@times, conj(model_g_f{nn, ii}), xtf), 3), [1 2 4 3]);
                    layer_responsef_padded(:,:,ii) = resizeDFT2(responsef, interp_sz);
                end
                responsef_padded(:,:,scale_ind)= sum(bsxfun(@times, layer_responsef_padded, permute(nweights,[3 1 2])), 3);
                response(nn,:,:,scale_ind) = ifft2(responsef_padded(:,:,scale_ind), 'symmetric');
            end
            [disp_row, disp_col, ~, peak_val] = resp_newton(permute(response(nn,:,:,:),[2 3 4 1]), responsef_padded, newton_iterations, ky, kx, use_sz);
            max_response(:,nn) = permute(peak_val, [3 2 1]); 
            sub_pos(:,:,nn) = bsxfun(@plus, [disp_row' disp_col']*currentScaleFactor*featureRatio, sub_init_pos(nn,:));
            % PSR and SCCM
            [psr(:,nn),peak_loc] = get_PSR(permute(response(nn,:,:,:),[2 3 4 1]), use_sz, nScales);
            for n = 1:nScales
                temp = (circshift(permute(response(nn,:,:,n),[2 3 4 1]), old_peak_loc(n,end:-1:1,nn)-peak_loc(n,end:-1:1))-old_response(:,:,n,nn)).^2;
                sccm(n, nn) = sum(temp(:));
            end
            % Weight_beta calculation
            sub_beta(:,nn) = psr(:,nn) + gamma./sccm(:, nn);
            
            for n = 1:nScales
                modify_response(:,:,n) = bsxfun(@times, fftshift(permute(response(nn,:,:,n),[2 3 4 1])), cos_mask(cos_window, peak_loc(n,:)));%
            end
            old_response(:,:,:,nn) = permute(response(nn,:,:,:),[2 3 4 1]);
            old_peak_loc(:,:,nn) = peak_loc;

            modify_response = bsxfun(@times,modify_response, permute(sub_beta(:,nn), [3 2 1]));
            sub_response(:,:,:,nn) = interp_res(modify_response, nScales, featureRatio);
        end

        % Joint response map
        corner_ur  = floor(sub_init_pos(4,:)./currentScaleFactor);
        corner_tl  = ceil(sub_init_pos(1,:)./currentScaleFactor);
        distance = abs(corner_ur - corner_tl);
        real_sz = floor(use_sz * featureRatio);
        
        im_res = zeros(corner_ur(1)-corner_tl(1) + real_sz(1), corner_ur(2)-corner_tl(2) + real_sz(2) ,nScales);
        
        im_res(1:real_sz(1), 1:real_sz(2),:)= im_res(1:real_sz(1), 1:real_sz(2),:) + sub_response(:,:,:,1);
        im_res(1:real_sz(1), distance(2)+1:end,:)=im_res(1:real_sz(1), distance(2)+1:end,:) + sub_response(:,:,:,2);
        im_res(distance(1)+1:end, 1:real_sz(2),:)= im_res(distance(1)+1:end, 1:real_sz(2),:) + sub_response(:,:,:,3);
        im_res(distance(1)+1:end, distance(2)+1:end,:) = im_res(distance(1)+1:end, distance(2)+1:end,:) + sub_response(:,:,:,4);
        im_res(floor(size(im_res,1)/2+(1:real_sz(1))-real_sz(1)/2), floor(size(im_res,2)/2+(1:real_sz(2))-real_sz(2)/2),:) = ...
            im_res(floor(size(im_res,1)/2+(1:real_sz(1))-real_sz(1)/2), floor(size(im_res,2)/2+(1:real_sz(2))-real_sz(2)/2),:) + sub_response(:,:,:,5);
        Vp = im_res; 

        [~,sc_index]= max(sum(max_response(:,:),2));
        
        % Structure comparison
        shift_vector = (permute(sub_pos(sc_index,:,:),[3,2,1]) - old_sub_pos);
        % Initial translation estimation
        pre_pos = sum(bsxfun(@times,shift_vector,(sub_beta(sc_index,:)./sum(sub_beta(sc_index,:)))'));
        % Reliable parts selection
        err = sqrt(shift_vector(:,1).^2+shift_vector(:,2).^2) - mean(sqrt(shift_vector(:,1).^2+shift_vector(:,2).^2));
        sigma_e = std(err,0);
        valid_part = find(abs(err) < sigma_e);
        
        % Initial scale estimation 
        pre_scale = mean(std(sub_pos(sc_index,:,valid_part), 0, 3)./(std(old_sub_pos(valid_part,:),0,1)));
        if isempty(valid_part) || pre_scale == Inf || isnan(pre_scale)
            pre_scale = 1;
        end
        
        % Update initial location and scale changes
        p = [size(Vp, 2)/2+pre_pos(2)./currentScaleFactor size(Vp, 1)/2+pre_pos(1)./currentScaleFactor base_target_sz(2) base_target_sz(1) 0];
       
        
        % Bayesian Inference framework
        param.est = affparam2mat([p(1), p(2), 1, p(5), p(4)/p(3), 0]);
        param.param = repmat(affparam2geom(param.est(:)), [1,n_sample])+ randn(6,n_sample).*repmat(affsig(:),[1,n_sample]);
        
        % Observation model calculation
        res = observation_score(Vp(:,:,sc_index), affparam2mat(param.param), base_target_sz, 1);
        
        % Save the previous PSR and SCCM
        results.sccm(frame,:) = sccm(sc_index,:);
        results.psr(frame,:) = psr(sc_index,:);
        
        % Update the final factor of scale changes 
        currentScaleFactor = currentScaleFactor * (scaleFactors(sc_index)*res.scale);
        
        % Adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % Update the final location 
        old_pos = pos;
        pos = pos + round((res.pos-[size(Vp, 1)/2 size(Vp, 2)/2])*currentScaleFactor); 
    end
    
    % Set current size of tracking object and sub-regions 
    target_sz = base_target_sz * currentScaleFactor;
    sub_target_sz = target_sz/2;
    
    % Location of sub-regions
    sub_init_pos(1,:) = pos - floor(sub_target_sz/2);
    sub_init_pos(2,:) = pos + [-floor(sub_target_sz(1)/2), floor(sub_target_sz(2)/2)];
    sub_init_pos(3,:) = pos + [floor(sub_target_sz(1)/2), -floor(sub_target_sz(2)/2)];
    sub_init_pos(4,:) = pos + floor(sub_target_sz/2);
    sub_init_pos(5,:) = pos;
    
    % Initialization
    if frame == 1
        
        old_peak_loc = ones(nScales, 2, subw_num);
        old_response = repmat(y, [1 1 nScales subw_num]);
        sub_net = repmat(permute((sub_init_pos - pos)', [3 1 2]), [nScales 1 1]);
        module = permute(sqrt(sub_net(:,1,:).^2+sub_net(:,2,:).^2), [1 3 2]);    
    end
    old_sub_pos = sub_init_pos;
    
    % Updating and training 
    for nn = 1:subw_num
        
        pixels = get_pixels(im,sub_init_pos(nn,:),round(sz*currentScaleFactor),sz);
        %xf = fft2(bsxfun(@times,get_features_CNN(pixels,use_sz,indLayers(ii)),cos_window));
        if (frame == 1)
            for ii = 1:numLayers
                xf = fft2(bsxfun(@times,get_features_CNN(pixels,use_sz,indLayers(ii)),cos_window));
                model_xf{nn, ii} = xf;
            end
            results.sccm(1,nn)=0;
        else
            update_res = GPR_update(results.psr, results.sccm, frame, nn, 50, 2); 
            if  update_res.res
                for ii = 1:numLayers
                    xf = fft2(bsxfun(@times,get_features_CNN(pixels,use_sz,indLayers(ii)),cos_window));
                    model_xf{nn, ii} = ((1 - learning_rate) * model_xf{nn, ii}) + (learning_rate * xf);
                end
            else
               results.sccm(frame, nn) = update_res.f;
            end
        end
        
        for ii = 1:numLayers
            % filter update of every Layer model
            g_f = single(zeros(size(model_xf{nn, ii})));
            h_f = g_f;
            l_f = g_f;
            mu    = 1;
            betha = 10;
            mumax = 10000;
            i = 1;
           
            T = prod(use_sz);
            S_xx = sum(conj(model_xf{nn, ii}) .* model_xf{nn, ii}, 3);
            params.admm_iterations = 2;
            %   ADMM
            while (i <= params.admm_iterations)
                %   solve for G- please refer to the paper for more details
                B = S_xx + (T * mu);
                S_lx = sum(conj(model_xf{nn, ii}) .* l_f, 3);
                S_hx = sum(conj(model_xf{nn, ii}) .* h_f, 3);
                g_f = (((1/(T*mu)) * bsxfun(@times, yf, model_xf{nn, ii})) - ((1/mu) * l_f) + h_f) - ...
                    bsxfun(@rdivide,(((1/(T*mu)) * bsxfun(@times, model_xf{nn, ii}, (S_xx .* yf))) - ((1/mu) * bsxfun(@times, model_xf{nn, ii}, S_lx)) + (bsxfun(@times, model_xf{nn, ii}, S_hx))), B);

                %   solve for H
                h = (T/((mu*T)+ params.admm_lambda))* ifft2((mu*g_f) + l_f);
                [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
                t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
                t(sx,sy,:) = h;
                h_f = fft2(t);

                %   update L
                l_f = l_f + (mu * (g_f - h_f));

                %   update mu- betha = 10.
                mu = min(betha * mu, mumax);
                i = i+1;
            end
            model_g_f{nn, ii} = g_f;
        end
    end
    
    time = time + toc();
    
    % Save position and calculate 
    rect_position(loop_frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];

    % Visualization 
    if visualization == 1 
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        im_to_show = double(im)/255;  
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if frame == 1
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            %axis off;
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            resp_sz = round(size(Vp)*currentScaleFactor*scaleFactors(scale_ind));
            xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
            ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            sc_ind = floor((nScales - 1)/2) + 1;
            
            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            resp_handle = imagesc(xs, ys, (Vp(:,:,sc_ind))); colormap hsv;
            alpha(resp_handle, 0.2);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(20, 30, ['# Frame : ' int2str(loop_frame) ' / ' int2str(num_frames)], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            text(20, 60, ['FPS : ' num2str(1/(time/loop_frame))], 'color', [1 0 0], 'BackgroundColor', [1 1 1], 'fontsize', 16);
            hold off;  
        end
        drawnow
        
    end
    loop_frame = loop_frame + 1;
end

% Save tracking resutls
fps = loop_frame / time;
results.type = 'rect';
results.res = rect_position;
results.fps = fps;
