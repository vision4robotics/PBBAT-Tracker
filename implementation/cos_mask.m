function mask = cos_mask(cos_window, peak_loc)
    cos_sz = size(cos_window);
    mask = zeros(cos_sz);
    peak_loc(1) = mod(peak_loc(1) - 1 + floor((cos_sz(2)-1)/2), cos_sz(2)) - floor((cos_sz(2)-1)/2);
    peak_loc(2) = mod(peak_loc(2) - 1 + floor((cos_sz(1)-1)/2), cos_sz(1)) - floor((cos_sz(1)-1)/2);
    xy_coordinate = peak_loc' + [1:cos_sz(2);1:cos_sz(1)]; 
    [~,valid_cos_x]=find(xy_coordinate(2,:)<=cos_sz(2) & xy_coordinate(2,:)>=1);
    [~,valid_cos_y]=find(xy_coordinate(1,:)<=cos_sz(1) & xy_coordinate(1,:)>=1);
    mask(xy_coordinate(2,valid_cos_x), xy_coordinate(1,valid_cos_y)) = cos_window(valid_cos_x, valid_cos_y);
end