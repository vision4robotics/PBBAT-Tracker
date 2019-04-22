function [psr, peak_location] = get_PSR(response, use_sz, nScales)
    [max_resp_row, max_row] = max(response, [], 1); 
    [init_max_response, max_col] = max(max_resp_row, [], 2); 
    max_row_perm = permute(max_row, [2 3 1]); 
    col = max_col(:)'; 
    row = max_row_perm(sub2ind(size(max_row_perm), col, 1:size(response,3)))';
    peak_location = [col' row];
    m = permute(init_max_response, [3 1 2]);
    avg = permute(mean(mean(response)),[3 1 2]);
    s = std(reshape(response, prod(use_sz),nScales),1)';
    psr = (m-avg)./s;
end