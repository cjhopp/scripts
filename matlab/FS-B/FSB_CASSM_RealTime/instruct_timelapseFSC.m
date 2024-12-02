function outstructTL = instruct_timelapseFSC(data)

load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/FSC_BaselineV1.mat')
load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/FSC_BaselineV1.mat')
load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/rejects.mat')

instruct.par = outstruct.par;
instruct.xsp = outstruct.S_Rpair(:,1);
instruct.ysp = outstruct.S_Rpair(:,2);
instruct.zsp = outstruct.S_Rpair(:,3);
instruct.xrp = outstruct.S_Rpair(:,4);
instruct.yrp = outstruct.S_Rpair(:,5);
instruct.zrp = outstruct.S_Rpair(:,6);
instruct.background = outstruct.slow{end,1};
instruct.kapX = 1;
instruct.kapY = 1;
instruct.kapZ = 1;
instruct.G    = outstruct.G{end,1};
instruct.regX = 1;
instruct.regY = 1;
instruct.regZ = 1;
instruct.nlres = 1e-6;


% data = [];
% K = 0;
% for k = 1:123
%     K = K + 1;
%     I = 0;
%     for i = [1:352 397:968 1013:1056]
%         I = I + 1;
%         data(I,K) = dsitempgathNov2021.th{i}(86,k);
%     end
% end


%% Data Quality

instruct.data = data';
badSources            = [((18-1)*44+1):18*44 ((23-1)*44+1):23*44];
rejects(badSources)   = [];
rjct = find(rejects == 1);
instruct.xsp(rjct,:)  = [];
instruct.ysp(rjct,:)  = [];
instruct.zsp(rjct,:)  = [];
instruct.xrp(rjct,:)  = [];
instruct.yrp(rjct,:)  = [];
instruct.zrp(rjct,:)  = [];
instruct.data(rjct,:) = [];
instruct.G(rjct,:)    = [];
instruct.Weight       = [];

[~,b] = size(instruct.G);
I = ones(instruct.par.nx,1)*1;
I(15:21) = 0;
I([15 21]) = 0.5;
I2 = [];
for i = 1:instruct.par.ny*instruct.par.nz
    I2 = [I2; I];
end
alpha = 1000;
instruct.PertMat = alpha*spdiags(I2,0,b,b);

outstructTL = tomo3DTTCR_timelapse(instruct);

end