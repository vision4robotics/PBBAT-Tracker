function res = GPR_update(psr_flow, sccm_flow, frame, sub_patch, mem_length, valid_region)

    res.res = false;
    res.f = 0;
    % hyperparameters for GPR
    sigma_f = 0.05;
    sigma_n = 0.001;
    l = 0.5;
    % Kernel function: SE kernel 
    SE = @(x,y)(sigma_f*exp(-abs(x-y).^2/(2*l)));
    
    length = size(psr_flow, 1);
    if length < mem_length + 3  
            mem_length = size(sccm_flow(3:end-1,sub_patch),1);
    end
    
    if frame <= 3 
        res.infer = true;
    else
        % Gaussian Process Regression
        inputs = double(psr_flow(end-mem_length:end,sub_patch));
        inputs = inputs - mean(inputs);
        y = sccm_flow(end-mem_length:end,sub_patch);
        y = y - mean(y);

        K = bsxfun(SE,inputs, inputs');
        L = chol(K(1:end-1,1:end-1)+sigma_n^2*eye(size(K,1)-1),'lower');

        alpha = L'\(L\y(1:end-1));
        fp = K(end,1:end-1)*alpha;
        % Calculate the mean and the variance  
        v = L\K(1:end-1,end);
        cov_fp = sqrt(K(end,end)-v'*v);

        if y(end) < fp+valid_region*cov_fp && y(end) > fp-valid_region*cov_fp
            res.res = true;
            res.f = max(res.f, fp+cov_fp*randn);
        end
    end        
end