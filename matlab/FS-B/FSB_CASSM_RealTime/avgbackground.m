load('/Volumes/FSC_Data/FSC_CASSM/tempgathers/May4.mat')

for i = 1:1056
    avgdtbackground(i) = mean(dsitempgath.th{i}(84,:));
end

save('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/avgdtbackground.mat','avgdtbackground')