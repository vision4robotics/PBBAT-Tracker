function res = observation_score(response, particles, target_size, nScales)
    
    loc = floor([particles(2,:)' particles(1,:)']);% [y x]
    sc  = particles(3,:)'; 
    sample_num = size(particles, 2); 
    sr  = [particles(6,:)' ones(sample_num, 1)];
    sz = floor(sc.*sr.*target_size(2)); 
    score = zeros(nScales, sample_num);
    delta_pos = zeros(2, sample_num);
    
    for warp_num = 1:sample_num
        
        xs = loc(warp_num,2) + (1:sz(warp_num, 2)) - floor(sz(warp_num, 2)/2);
        ys = loc(warp_num,1) + (1:sz(warp_num, 1)) - floor(sz(warp_num, 1)/2);
        
        wimg1 = response(ys(ys>=1 & ys<=size(response,1)), xs(xs>=1 & xs<=size(response,2)), :);
        wimg1(wimg1(:)<0) = eps;
        
        score(:, warp_num) = sum(sum(wimg1, 1), 2)./prod(sz(warp_num, :));
    end
    [~, sc_ind] = max(sum(score, 2)); 
    res.pos = sum((loc).*(score(sc_ind,:).^2)')/sum(score(sc_ind,:).^2);
     
    res.scale = sum(sc.*score(sc_ind,:)')/sum(score(sc_ind,:));

end