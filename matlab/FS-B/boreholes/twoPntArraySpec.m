  function [xl,yl,zl] = twoPntArraySpec(loc1,loc2,wlDepth,elNum,delta)

% function [xl,yl,zl] = twoPntArraySpec(loc1,loc2,wlDepth,elNum,delta)
%
%
% Input arguments
% --------------------
% 1. wellhead location (x,y,z) : location of wellhead
% 2. TD location (x,y,z) : location of toe of the well (assuming straight)
% 3. wlDepth : wireline depth (m) of the start of the downhole array
% 4. elNum   : number of elements in the array
% 5. delta   : spacing between array elements
%
% Output Variables
% --------------------
%
%
%

% plot flag
graphicsOn = 0;

% base vectors
xl = zeros(1,elNum);
yl = zeros(1,elNum);
zl = zeros(1,elNum);

totalDepth = sqrt((loc2(1)-loc1(1))^2 + (loc2(2)-loc1(2))^2 + (loc2(3)-loc1(3))^2);

% creating wireline depths for the array
wlVec = wlDepth + [0:1:(elNum-1)].*delta;


locs = [0,0,0];
loce = [loc2(1)-loc1(1),loc2(2)-loc1(2),loc2(3)-loc1(3)];

% interpolating wireline locations to xl/yl/zl
xl = interp1([0,totalDepth],[loc1(1),loc2(1)],wlVec);
yl = interp1([0,totalDepth],[loc1(2),loc2(2)],wlVec);
zl = interp1([0,totalDepth],[loc1(3),loc2(3)],wlVec);

% plotting for QC
if(graphicsOn)
    plot3([loc1(1),loc2(1)],[loc1(2),loc2(2)],[loc1(3),loc2(3)],'r-'); hold on;
    plot3([loc1(1),loc2(1)],[loc1(2),loc2(2)],[loc1(3),loc2(3)],'ro');
    plot3(xl,yl,zl,'bo');
    axis xy;
    axis equal;
    hold off;
end;


