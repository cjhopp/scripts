rawdatapath     = '/Volumes/Tanner_FSC/FSC_CASSM/FSB_CASSM_Geodes_2023';
workingdatapath = '/Volumes/Tanner_FSC/FSC_CASSM/CASSMdata';


newfolderlist = newfoldercheck(rawdatapath,workingdatapath);

Lnew = length(newfolderlist);

for i = 1:Lnew
    disp(['checking new folder ' newfolderlist{i} ' (' num2str(i) '/' num2str(Lnew) '):'])
    folderpath = [rawdatapath '/' newfolderlist{i}];
    foldersize = 0;
    while foldersize ~= 33580
        disp('    waiting for folder download completely...')
        files = dir([folderpath '/*.dat']);
        size  = sum([files(1:end).bytes])/1000;
        pause(10)
    end
    movefile(folderpath,workingdatapath);
    disp('...new moving folder to working directory')
    disp(' ')
end