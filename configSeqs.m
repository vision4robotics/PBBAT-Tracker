function seqs=configSeqs(video_path_UAV123)
dataSet = video_path_UAV123;

seqUAV123_10fps = {
    struct('name','person7_2','path',[dataSet '\person7\'],'startFrame',417,'endFrame',689,'nz',6,'ext','jpg','init_rect',[0,0,0,0]),...
    };
seqs = seqUAV123_10fps;