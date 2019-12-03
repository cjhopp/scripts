function output=stereogrid(angle,projection)
% output=stereogrid(angle,projection)
% plot the grid of for stereographic projection
% angle: angle between each gridline (default=10)
% projection: 'equalangle' or 'equalarea'
% output: vector of handles on lines
%
% modified code from Gerard V. Middleton
% in: Data analysis in the earth sciences using Matlab(R)
% Prentice Hall, 2000. ISBN 0-13-393505-1
if nargin==0
    angle=10;
    projection='equalangle';
elseif nargin==1;
    projection='equalangle';
end
output=[];
a1=gca;
axis off
hold on
if strcmp(projection,'equalangle') % net for equal angle
    N = 50;
    psi = [0:180/N:180];
    
    for i = 0:(90-angle)/angle
        rdip = i*(angle);                             
        radip = atand(tand(rdip)*sind(psi));
        rproj = tand((90 - radip)/2);
        x1 = rproj .* sind(psi);
        x2 = rproj .* (-sind(psi));
        y = rproj .* cosd(psi);
        output=[output; plot(x1,y,':k',x2,y,':k')];
    end
    for i = 0:(90-angle)/angle
        alpha = i*(angle);
        xlim = sind(alpha);
        ylim = cosd(alpha);
        x = [-xlim:0.01:xlim];
        d = 1/cosd(alpha);
        rd = d*sind(alpha);
        y0 = sqrt(rd*rd - (x .* x));
        y1 = d - y0;
        y2 = - d + y0;
        output=[output; plot(x,y1,':k',x,y2,':k')];
    end
end
if strcmp(projection,'equalarea') % net for equal area
    N = 60;
    psi = [0:180/N:180];

    for i = 0:(90-angle)/angle                       %plot great circles
        rdip = i*(angle);                            
        radip = atand(tand(rdip)*sind(psi));
        rproj = sqrt(2)*sind((90 - radip)/2);
        x1 = rproj .* sind(psi);
        x2 = rproj .* (-sind(psi));
        y = rproj .* cosd(psi);
        output=[output; plot(x1,y,':k',x2,y,':k')];
    end
    for i = 0:(90-angle)/angle                       %plot small circles
        alpha = i*(angle);
        xlim = sind(alpha);
        ylim = cosd(alpha);
        x = [-xlim:0.01:xlim];
        d = 1/cosd(alpha);
        rd = d*sind(alpha);
        y0 = sqrt(rd*rd - (x .* x));
        y1 = d - y0;
        y2 = - d + y0;
        output=[output; plot(x,y1,':k',x,y2,':k')];
    end
end

output=[output; plot([0 0],[-1 1],':k',[-1 1],[0 0],':k')];
axis('square');