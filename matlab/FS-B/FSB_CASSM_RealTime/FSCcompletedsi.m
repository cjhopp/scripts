function FSCcompletedsi(DIR)
DIR = uigetdir();
list = textscan(ls(DIR),'%f');
list = strtrim(convertStringsToChars(string(list{1,1})));

L = length(list);
E = zeros(L,1);

parfor i = 1:L
    if ~exist([[DIR '/' list{i} '/'] 'dsiDataV1.mat'],'file')
        try
            disp(cat(1,[':Preparing data in folder ',list{i},' (',num2str(i),' of ',num2str(L),')']));
            dsiData = fscCASSMdirRead([DIR '/' list{i} '/']);
            fsbpreprocessing([DIR '/' list{i} '/'], dsiData)
        catch
            E(i) = 1;
        end
    end
end

E = logical(E);

% for i = 1:L
%     if E(i)
%         eval(['movefile ' DIR '/' list{i} ' /Users/tanner/Downloads/corrupt'])
%     end
% end
end