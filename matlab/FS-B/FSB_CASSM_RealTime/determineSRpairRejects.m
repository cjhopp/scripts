% load('/Volumes/FSC_Data/FSC_CASSM/tempgathers/May2and3.mat')

%bad sources are 18 and 23 for FSC2023
badSources = [((18-1)*44+1):18*44 ((23-1)*44+1):23*44];
L          = 1056;
traces     = 1:L;
VAR        = zeros(L,1);
for i = traces
    dsitempgath.th{i}(84,:) = dsitempgath.th{i}(84,:) - mean(dsitempgath.th{i}(84,:));
    VAR(i) = var(dsitempgath.th{1,i}(84,:));
end


P           = round(0.15*(L-length(badSources)));
badtraces   = sort(VAR,1, 'descend');
badtraceidx = ismember(VAR,badtraces(1:P));

rejects              = zeros(1056,1);
rejects(badSources)  = 1;
rejects(badtraceidx) = 1;

save('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/rejects.mat','rejects')