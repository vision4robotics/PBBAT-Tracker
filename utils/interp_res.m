function Vp = interp_res(im_res, nScales, featureRatio)
    [Xi, Yi] = meshgrid(linspace(1,size(im_res, 2),floor(size(im_res, 2)*featureRatio)), linspace(1,size(im_res, 1),floor(size(im_res, 1)*featureRatio)));
    for n = 1:nScales
        Vp(:,:,n) = interp2(im_res(:,:,n), Xi, Yi);
    end
end