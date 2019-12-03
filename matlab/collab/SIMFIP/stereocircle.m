function output=stereocircle(data,projection,width,color)
% output=stereocircle(data,projection,width,color)
% plot greatcircle for a given azimuth/dip data
% data: azimuth/dip Nx2 matrix
% projection: 'equalangle' or 'equalarea'. Default 'equalangle'
% width: line width. Default 1.
% color: line color [r g b]. default [0 0 0]
%
% modified code from Gerard V. Middleton
% in: Data analysis in the earth sciences using Matlab(R)
% Prentice Hall, 2000. ISBN 0-13-393505-1
if nargin==1
    projection='equalangle';
    width=1;
    color=[0 0 0];
elseif nargin==2
    width=1;
    color=[0 0 0];
elseif nargin==3
    color=[0 0 0];
elseif nargin<=0
    error('not enough input arguments')
end

if isempty(data); output=[]; return; end;

output=[];
a1=gca;
axis off
hold on
if strcmp(projection,'equalangle') % net for equal angle
    N = 60;
    psi = [0:180/N:180];
    
    for i = 1:length(data(:,1))
        rdip = data(i,2);                             
        radip = atand(tand(rdip)*sind(psi));
        rproj = tand((90 - radip)/2);
        x = rproj .* sind(psi);
        y = rproj .* cosd(psi);
        if rdip==90;
            x=[0;0];
            y=[1;-1];
        end
        for j=1:length(x)
            t=[cosd(-data(i,1)+90) -sind(-data(i,1)+90); sind(-data(i,1)+90) cosd(-data(i,1)+90)]*[x(j); y(j)];
            x(j)=t(1); y(j)=t(2);
        end
        output=[output; plot(x,y,'Color',color,'LineWidth',width)];
    end
end
if strcmp(projection,'equalarea') % net for equal area
    N = 50;
    psi = [0:180/N:180];

    for i = 1:length(data(:,1))                       %plot great circles
        rdip = data(i,2);                            
        radip = atand(tand(rdip)*sind(psi));
        rproj = sqrt(2)*sind((90 - radip)/2);
        x = rproj .* sind(psi);
        y = rproj .* cosd(psi);
        if rdip==90;
            x=[0;0];
            y=[1;-1];
        end
        for j=1:length(x)
            t=[cosd(-data(i,1)+90) -sind(-data(i,1)+90); sind(-data(i,1)+90) cosd(-data(i,1)+90)]*[x(j); y(j)];
            x(j)=t(1); y(j)=t(2);
        end
        output=[output; plot(x,y,'Color',color,'LineWidth',width)];
    end
end

axis('square');