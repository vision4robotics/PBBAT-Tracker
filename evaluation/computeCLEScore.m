function CLE = computeCLEScore(bb1,bb2)
    center1 = computeCenter(bb1);
    center2 = computeCenter(bb2);
    CLE = sqrt(sum((center1-center2).^2,2));
end