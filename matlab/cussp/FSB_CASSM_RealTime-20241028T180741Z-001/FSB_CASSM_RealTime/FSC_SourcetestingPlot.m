close all
load('/Volumes/Tanner_FSC/FSC_CASSM/May2data/20230502072550/dsiDataV1.mat')
figure
    for i = 1:24
        subplot(4,6,i)
            plotgather(dsiDataV1,i,1:44,100)
            ylim([0 30])
            title(['S' num2str(i)])
    end
    
load('/Volumes/Tanner_FSC/FSC_CASSM/CASSMdata/20230502085109/dsiDataV1.mat');
figure
    for i = 1:24
        subplot(4,6,i)
            plotgather(dsiDataV1,i,45:80,0)
            ylim([0 30])
            title(['S' num2str(i)])
    end