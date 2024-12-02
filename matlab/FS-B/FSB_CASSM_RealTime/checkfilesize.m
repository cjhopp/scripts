
DIR = uigetdir();

list = textscan(ls(DIR),'%f');
list = sort(strtrim(convertStringsToChars(string(list{1,1}))));
L    = length(list);

for i = 1:L
    files = dir([DIR '/' list{i} '/*.dat']);
    size(i) = sum([files(1:end).bytes])/1000;
end

round(mean(size(1:187)))

figure
    histogram(size(1:end),100)

figure
    plot(size)