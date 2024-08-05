function [newfolderlist] = newfoldercheck(rawdatapath,workingdatapath)
    
    rawlist = textscan(ls(rawdatapath),'%f');
    rawlist = sort(strtrim(convertStringsToChars(string(rawlist{1,1}))));
    L_raw   = length(rawlist);
    
    for i = 1:L_raw
        folderdate(1)  = str2double(rawlist{i}(1:4));
        folderdate(2)  = str2double(rawlist{i}(5:6));
        folderdate(3)  = str2double(rawlist{i}(7:8));
        folderdate(4)  = str2double(rawlist{i}(9:10));
        folderdate(5)  = str2double(rawlist{i}(11:12));
        folderdate(6)  = str2double(rawlist{i}(13:14));
        folderdates(i) = datenum(folderdate);
    end
    
    worklist = textscan(ls(workingdatapath),'%f');
    worklist = sort(strtrim(convertStringsToChars(string(worklist{1,1}))));
    L_work   = length(worklist);
    
    % finding the last working epoch. 
    lastfolder(1) = str2double(worklist{end}(1:4));
    lastfolder(2) = str2double(worklist{end}(5:6));
    lastfolder(3) = str2double(worklist{end}(7:8));
    lastfolder(4) = str2double(worklist{end}(9:10));
    lastfolder(5) = str2double(worklist{end}(11:12));
    lastfolder(6) = str2double(worklist{end}(13:14));
    lastfolder    = datenum(lastfolder);
    
    lastidx = find(folderdates == lastfolder);
    
    if lastidx == L_raw
        newfolderlist = [];
    else 
        newfolderlist = rawlist((lastidx+1):L_raw);
    end


end