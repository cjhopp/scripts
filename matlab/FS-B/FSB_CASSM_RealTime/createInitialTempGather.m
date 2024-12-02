function dsitempgath = createInitialTempGather(DIR,savedir)
% FSCcompletedsi(DIR);


list = textscan(ls(DIR),'%f');
list = sort(strtrim(convertStringsToChars(string(list{1,1}))));
L    = length(list);
dsitempgath.th  = [];
dsitempgath.dat = [];
srcs = 24;
recs = 44;

for i = 1:(srcs*recs)
    dsitempgath.dat{i} = zeros(3840,L);
    dsitempgath.th{i} = zeros(84,L);
end

for i = 1:L
    disp(['Adding epoch number ' num2str(i) ' to temp-gather'])
    if exist([DIR '/' list{i} '/dsiDataV1.mat'],'file')
        load([DIR '/' list{i} '/dsiDataV1.mat'])
        for j = 1:(srcs*recs)
            [S,R] = TankIndexNum2SrcRec(j,recs);
            try
            dsitempgath.dat{j}(:,i) = dsiDataV1.dat{S}(1:3840,R);
            dsitempgath.th{j}(:,i) = dsiDataV1.th{S}(1:84,R);
            catch
            end
        end
    end
end

[D,L]  = size(dsitempgath.dat{1});
fh{1,1}  = L;
fh{1,2}  = []; 
fh{1,3}  = []; 
fh{1,4}  = []; 
fh{1,5}  = []; 
fh{1,6}  = []; 
fh{1,7}  = D; 
fh{1,8}  = 1./48000; 
fh{1,9}  = 0;
fh{1,10} = 0.025;
fh{1,11} = [];
fh{1,12} = srcs*recs;
fh{1,13} = L;

dsitempgath.fh = fh;

N = dsitempgath.fh{12};
L = dsitempgath.fh{1};

for j = 1:N
    for i = 1:L
        if all(dsitempgath.dat{j}(:,i) == 0)
            dsitempgath.th{j}(6,i) = -1;
        end
    end
end

% 
% winsize  = 0.003;
% wintaper = winsize/10;
% load('/Users/tanner/Rice Geophysics Dropbox/Tanner Shadoan/fsbCASSM/TASscripts/FSB_CASSM_RealTime/TravelTimePicksFSC.mat')
% for gath = 1:recs*srcs
%     disp(['Now working on ' num2str(gath) '.'])
%     pick  = data(gath,1);
%     if pick ~= 0
%         t0 = pick + winsize/2;
%         [dsitempgath]                                           = dsiCASSMdelaySequenceEstWinPar(dsitempgath,t0,wintaper,winsize,84,dsitempgath.fh{8}/1000,gath,1,1/10000);
%         [dsitempgath.th{gath}(81,:),dsitempgath.th{gath}(82,:)] = dsi_rmsAmpWin(dsitempgath,t0,winsize,wintaper,gath);
%         [~,~,~,dsitempgath.th{gath}(83,:),~,~]                  = dsi_centfreqWin(dsitempgath,t0,winsize,wintaper,gath);
%         dsitempgath.th{gath}(83,:)                              = dsitempgath.th{gath}(83,:)/1000;
%     end
% end

% save([savedir 'tempgather.mat'],'dsitempgath','-v7.3')

end %function