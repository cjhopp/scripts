% testWellGeomFSB_V1.m
% 
% Test well geometries for FSB : version 1
%              



%               1                   2                   3                      
% FSB holes	Easting (m)     Northing (m)	Collar elevation (m)      	
SIG01a   =       [2579345.22   1247580.39      513.2];         %    BFS B1 20181116
SIG01b   =       [2579345.22   1247580.39      450.00];

SIG02a   =       [2579334.48    1247570.98       513.2];       %    BFS B2 inclined 20181120
SIG02b   =       [2579329.36    1247577.86       460.41];

SIG03a   =       [2579324.91    1247611.68       514.13];      %    BFS B3 20181119
SIG03b   =       [2579322.61    1247556.79       449.53];

SIG04a   =       [2579325.5     1247612.05       514.07];      %    BFS B4 20181119
SIG04b   =       [2579338.71    1247569.11       447.96];

SIG05a   =       [2579332.57    1247597.29       513.78];      %    BFS B5 20181119
SIG05b   =       [2579321.52    1247556.01       473.52];     

SIG06a   =       [2579334.35    1247598.44       513.72];      %    BFS B6
SIG06b   =       [2579338.50    1247569.01       473.70];

SIG07a   =       [2579336.22    1247599.75       513.76];      %    BFS B7
SIG07b   =       [2579351.79    1247579.12       474.15]; 

% % Variant (a)
% SIG08a   =       [2579333.0     1247603.00       513.78];      %    BFS B8, AE1 above fault
% SIG08b   =       [2579340.0     1247570.00       463.41];
% 
% SIG09a   =       [2579330.00    1247607.00       513.2];       %    BFS B9, AE2
% SIG09b   =       [2579325.00    1247565.00       463.41];
% %%%%%%%%%%%%%


% % Short variant
SIG08a   =       [2579334.0     1247602.00       513.78];      %    BFS B8, AE1 above fault
SIG08b   =       [2579326.5     1247563.50       472.50];       %  r - -

SIG09a   =       [2579328.00    1247609.00       513.2];       %    BFS B9, AE2 below fault
SIG09b   =       [2579335.0     1247570.00       458.0];      %  r ---
% %%%%%%%%%%%%%

% Hydrophones & CASSM
figure;
% at present, X is easting, Y is northing, Z is elevation
plot3([SIG01a(1),SIG01b(1)],[SIG01a(2),SIG01b(2)],[SIG01a(3),SIG01b(3)],'k-'); hold on;
plot3([SIG02a(1),SIG02b(1)],[SIG02a(2),SIG02b(2)],[SIG02a(3),SIG02b(3)],'k--'); 

plot3([SIG03a(1),SIG03b(1)],[SIG03a(2),SIG03b(2)],[SIG03a(3),SIG03b(3)],'g-'); % lower - 24 hydrophones
plot3([SIG04a(1),SIG04b(1)],[SIG04a(2),SIG04b(2)],[SIG04a(3),SIG04b(3)],'g--');

plot3([SIG05a(1),SIG05b(1)],[SIG05a(2),SIG05b(2)],[SIG05a(3),SIG05b(3)],'b-'); % upper - 8 cassm sources
plot3([SIG06a(1),SIG06b(1)],[SIG06a(2),SIG06b(2)],[SIG06a(3),SIG06b(3)],'b--');
plot3([SIG07a(1),SIG07b(1)],[SIG07a(2),SIG07b(2)],[SIG07a(3),SIG07b(3)],'b-.');

%% As built locations

shiz = [2579324.49687 1247592.9675 491.9781;
        2579324.47482 1247591.3557 490.0671;
        2579324.45196 1247589.7436 488.1565;
        2579324.42866 1247588.132 486.2454;
        2579324.40549 1247586.5206 484.3342;
        2579324.38464 1247584.9097 482.4225;
        2579324.36524 1247583.3001 480.5097;
        2579324.34529 1247581.6901 478.5972;
        2579324.32441 1247580.0796 476.6852;
        2579324.30444 1247578.4679 474.7741;
        2579324.28608 1247576.8557 472.8636;
        2579324.2671 1247575.2429 470.9534;
        2579324.2482 1247573.6299 469.0435;
        2579324.22945 1247572.0169 467.1336;
        2579324.20813 1247570.4047 465.223;
        2579324.18595 1247568.7916 463.3131;
        2579324.16327 1247567.1781 461.4037;
        2579324.14131 1247565.5641 459.4946;
        2579324.11819 1247563.9509 457.5849;
        2579324.09117 1247562.3397 455.6735;
        2579324.06204 1247560.7305 453.7606;
        2579324.0327 1247559.1223 451.8467;
        2579329.17688 1247597.8316 492.0198;
        2579329.55651 1247596.495 489.9415;
        2579329.93608 1247595.1581 487.8633;
        2579330.31607 1247593.8213 485.7852;
        2579330.69621 1247592.4842 483.7073;
        2579331.07555 1247591.1471 481.6292;
        2579331.45355 1247589.8114 479.5501;
        2579331.83095 1247588.475 477.4713;
        2579332.20912 1247587.1386 475.3925;
        2579332.58727 1247585.8011 473.3145;
        2579332.96631 1247584.4626 471.2373;
        2579333.34493 1247583.1243 469.16;
        2579333.72263 1247581.7854 467.0827;
        2579334.09847 1247580.4462 465.0054;
        2579334.47514 1247579.1063 462.9287;
        2579334.85297 1247577.7667 460.852;
        2579335.23231 1247576.4266 458.7759;
        2579335.6125 1247575.0857 456.7004;
        2579335.9923 1247573.7446 454.625;
        2579336.3728 1247572.4026 452.5504;
        2579336.7543 1247571.0595 450.4766;
        2579337.1362 1247569.7158 448.4033;
        2579324.46129 1247590.3884 488.9207;
        2579324.12792 1247564.5962 458.3487;
        2579329.86015 1247595.4255 488.279;
        2579335.9162 1247574.0129 455.0401;
        2579328.42237 1247584.2444 500.604;
        2579321.582 1247563.2318 478.8632;
        2579334.79024 1247589.14718 501.6884;
        2579337.27833 1247570.6516 476.9351;
        2579340.15296 1247593.54974 503.041;
        2579351.2349 1247579.5168 477.7178];



%% Theoretical locations
axis equal;
% [xl3,yl3,zl3] = twoPntArraySpec(SIG03a,SIG03b,24,24,2.5);
% [xl4,yl4,zl4] = twoPntArraySpec(SIG04a,SIG04b,22,24,2.5);
% plot3(xl3,yl3,zl3,'g<');
% plot3(xl4,yl4,zl4,'go');
% 
% [xl5,yl5,zl5] = twoPntArraySpec(SIG05a,SIG05b,20,8,4.5);
% [xl6,yl6,zl6] = twoPntArraySpec(SIG06a,SIG06b,16,8,4.5);
% [xl7,yl7,zl7] = twoPntArraySpec(SIG07a,SIG07b,14,8,4.5);
% plot3(xl5,yl5,zl5,'r<');
% plot3(xl6,yl6,zl6,'ro');
% plot3(xl7,yl7,zl7,'r*');
plot3(shiz(:,1), shiz(:,2), shiz(:,3), 'r*');
xlabel('Easting (m)');
ylabel('Northing (m)');
zlabel('Elevation (m)');


legend('B1','B2','B3','B4','B5','B6','B7','B3 hydro','B4 hydro','B5 CASSM','B6 CASSM','B7 CASSM');
grid on;

% MEQ array
figure;
% at present, X is easting, Y is northing, Z is elevation
plot3([SIG01a(1),SIG01b(1)],[SIG01a(2),SIG01b(2)],[SIG01a(3),SIG01b(3)],'k-', 'LineWidth', 2); hold on;
plot3([SIG02a(1),SIG02b(1)],[SIG02a(2),SIG02b(2)],[SIG02a(3),SIG02b(3)],'k--', 'LineWidth', 2); 

plot3([SIG03a(1),SIG03b(1)],[SIG03a(2),SIG03b(2)],[SIG03a(3),SIG03b(3)],'g-', 'LineWidth', 2); % lower - 24 hydrophones
plot3([SIG04a(1),SIG04b(1)],[SIG04a(2),SIG04b(2)],[SIG04a(3),SIG04b(3)],'g--', 'LineWidth', 2);

plot3([SIG05a(1),SIG05b(1)],[SIG05a(2),SIG05b(2)],[SIG05a(3),SIG05b(3)],'b-', 'LineWidth', 2); % upper - 8 cassm sources
plot3([SIG06a(1),SIG06b(1)],[SIG06a(2),SIG06b(2)],[SIG06a(3),SIG06b(3)],'b--', 'LineWidth', 2);
plot3([SIG07a(1),SIG07b(1)],[SIG07a(2),SIG07b(2)],[SIG07a(3),SIG07b(3)],'b-.', 'LineWidth', 2);

plot3([SIG08a(1),SIG08b(1)],[SIG08a(2),SIG08b(2)],[SIG08a(3),SIG08b(3)],'r--', 'LineWidth', 2);
plot3([SIG09a(1),SIG09b(1)],[SIG09a(2),SIG09b(2)],[SIG09a(3),SIG09b(3)],'r-', 'LineWidth', 2);

axis equal;


[xl3,yl3,zl3] = twoPntArraySpec(SIG03a,SIG03b,34,2,40);
[xl4,yl4,zl4] = twoPntArraySpec(SIG04a,SIG04b,32,2,40);
plot3(xl3,yl3,zl3,'go', 'LineWidth', 2, 'MarkerSize', 20);
plot3(xl4,yl4,zl4,'go', 'LineWidth', 2, 'MarkerSize', 20);

[xl5,yl5,zl5] = twoPntArraySpec(SIG05a,SIG05b,20,2,30);
[xl6,yl6,zl6] = twoPntArraySpec(SIG06a,SIG06b,16,2,30);
[xl7,yl7,zl7] = twoPntArraySpec(SIG07a,SIG07b,14,2,30);
plot3(xl5,yl5,zl5,'bo', 'LineWidth', 2, 'MarkerSize', 20);
plot3(xl6,yl6,zl6,'bo', 'LineWidth', 2, 'MarkerSize', 20);
plot3(xl7,yl7,zl7,'bo', 'LineWidth', 2, 'MarkerSize', 20);

%twoPntArraySpec(loc1, loc2, wlDepth, elNum, delta)
[xl8,yl8,zl8] = twoPntArraySpec(SIG08a,SIG08b,25,8,4);
[xl9,yl9,zl9] = twoPntArraySpec(SIG09a,SIG09b,35,8,4);
plot3(xl8,yl8,zl8,'r*', 'LineWidth', 2, 'MarkerSize', 20);
plot3(xl9,yl9,zl9,'r*', 'LineWidth', 2, 'MarkerSize', 20);
xlabel('Easting (m)');
ylabel('Northing (m)');
zlabel('Elevation (m)');

[xl3,yl3,zl3] = twoPntArraySpec(SIG03a,SIG03b,24,24,2.5);
[xl4,yl4,zl4] = twoPntArraySpec(SIG04a,SIG04b,22,24,2.5);
plot3(xl3,yl3,zl3,'gs', 'LineWidth', 2, 'MarkerSize', 10);
plot3(xl4,yl4,zl4,'gs', 'LineWidth', 2, 'MarkerSize', 10);


legend('B1 SIMFIP','B2 (injection)','B3','B4','B5','B6','B7','B8', 'B9', 'B3 3C','B4 3C','B5 3C','B6 3C','B7 3C','B8 AE','B9 AE');

grid on;


