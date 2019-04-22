function center = computeCenter(bb)
    center = [bb(:,1)+bb(:,3) bb(:,2)+bb(:,4)]/2;
end