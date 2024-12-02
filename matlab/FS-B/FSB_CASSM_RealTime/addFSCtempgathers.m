function [dsitempgathout,N,L] = addFSCtempgathers(DIR,dsitempgathin)

%outputs
% N represents the number of epochs in dsitempgath input
% L represents the number of epochs in dsitempgath output

list = textscan(ls(DIR),'%f');
list = sort(strtrim(convertStringsToChars(string(list{1,1}))));

L = length(list);
srcs = 24;
recs = 44;

for i = 1:(srcs*recs)

end

N = dsitempgathin.fh{1};

for i = (N+1):L
    disp(['Adding epoch number ' num2str(i) ' to temp-gather'])
    if exist([DIR '/' list{i} '/dsiDataV1.mat'],'file')
        load([DIR '/' list{i} '/dsiDataV1.mat'])
        for j = 1:(srcs*recs)
            [S,R] = IndexNum2SrcRec(j);
            try
                dsitempgathin.dat{j}(:,i) = dsiDataV1.dat{S}(1:3840,R);
                dsitempgathin.th{j}(:,i)  = dsiDataV1.th{S}(1:84,R);
            catch
            end
        end
    else
        for j = 1:(srcs*recs)
            dsitempgathin.dat{j}(:,i) = zeros(3840,1);
            dsitempgathin.th{j}(:,i)  = zeros(84,1);
        end
    end
end

[D,L]  = size(dsitempgathin.dat{1});
fh{1,1}  = L;
fh{1,2}  = []; 
fh{1,3}  = []; 
fh{1,4}  = []; 
fh{1,5}  = []; 
fh{1,6}  = []; 
fh{1,7}  = D; 
fh{1,8}  = 2.0833e-05; 
fh{1,9}  = 0;
fh{1,10} = D*fh{1,8};
fh{1,11} = [];
fh{1,12} = srcs*recs;
fh{1,13} = L;

dsitempgathin.fh = fh;

dsitempgathout = dsitempgathin;