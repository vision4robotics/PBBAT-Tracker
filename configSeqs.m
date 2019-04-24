function seqs=configSeqs(video_path_UAV123)
dataSet = video_path_UAV123;

seqUAV123_10fps = {
    struct('name','group3_2','path',[dataSet '\group3\'],'startFrame',523,'endFrame',943,'nz',6,'ext','jpg','init_rect',[0,0,0,0]),...
    };
seqs = seqUAV123_10fps;