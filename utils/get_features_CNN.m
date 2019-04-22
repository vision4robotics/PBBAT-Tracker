function xf = get_features_CNN(im,use_sz,indLayer)
    
    global net
    global enableGPU

    if isempty(net)
        initial_net();
    end
    
    % Preprocessing
    img = single(im);        % note: [0, 255] range
    img = imResample(img, net.meta.normalization.imageSize(1:2));
    
    average=net.meta.normalization.averageImage;

    if numel(average)==3
        average=reshape(average,1,1,3);
    end

    img = bsxfun(@minus, img, average); %均值归零

    if enableGPU, img = gpuArray(img); end
    
    % Run the CNN
    res = vl_simplenn(net,img);
    
    
    if enableGPU
        x = gather(res(indLayer).x);
    else
        x = res(indLayer).x;
    end
    % 将原始大小的图像resize成cell化后的图像,Fourier
    xf = imResample(x, use_sz(1:2));
    
end