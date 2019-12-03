close all
clear all
clc

Test_372=dlmread('TERRIBFS372m_Processed10Hz1222016.txt');

Pressure10Hz=Test_372(:,1);
Debit10Hz=Test_372(:,2);
Northern10Hz = Test_372(:,4);
Western10Hz = Test_372(:,5)*(-1); %Attention
Vertical10Hz = Test_372(:,6);
Time10Hz=[0:length(Pressure10Hz)-1]'/10;


% cut data out
ix1=33490;
ix2=90009;


Pressure10Hz=Pressure10Hz(ix1:ix2);
Debit10Hz=Debit10Hz(ix1:ix2);
Western10Hz=Western10Hz(ix1:ix2);
Northern10Hz=Northern10Hz(ix1:ix2);
Vertical10Hz=Vertical10Hz(ix1:ix2);
Time10Hz=Time10Hz(ix1:ix2);

% bring time and displacement to 0 / origin at the start of the serie
Time10Hz=Time10Hz-Time10Hz(1);
Western10Hz=Western10Hz-Western10Hz(1);
Northern10Hz=Northern10Hz-Northern10Hz(1);
Vertical10Hz=Vertical10Hz-Vertical10Hz(1);

We_vec=Western10Hz(2:end)-Western10Hz(1:end-1);
No_vec=Northern10Hz(2:end)-Northern10Hz(1:end-1);
Ve_vec=Vertical10Hz(2:end)-Vertical10Hz(1:end-1);

m=[Pressure10Hz(1:end-1),We_vec,No_vec,Ve_vec];

v=[
        m(:,2)'
        m(:,3)'
        m(:,4)'
        ];
    
    %------------linear interpolation
SS=[];
for j=1:length(v(1,:)); 
    S(j)=norm(v(:,j)); 
end;
SS=[SS S];


c_sum=cumsum(SS);
correct_distance = sum(c_sum)/length(v(:,1));

y1=Western10Hz(1:56519);
y2=Northern10Hz(1:56519);
y3=Vertical10Hz(1:56519);

xi=1:0.5:725;
yi_1=interp1(c_sum,y1,xi);
yi_2=interp1(c_sum,y2,xi);
yi_3=interp1(c_sum,y3,xi);


figure
plot3(y1 , y2 , y3 , '-k'); hold on;
plot3(yi_1, yi_2, yi_3, 'or'); hold on;
grid on
axis equal
xlabel('west')
ylabel ('north')
zlabel ('vertical')

mi=[yi_1', yi_2', yi_3'];

ve=[
        mi(:,1)'
        mi(:,2)'
        mi(:,3)'
        ];
    
west=yi_1';
north=yi_2';
vert=yi_3';

% Displacement in 3D
x_norm=north(2:end)-north(1:end-1);
y_norm=west(2:end)-west(1:end-1);
z_norm=vert(2:end)-vert(1:end-1);

n_dp =[ x_norm y_norm z_norm];
for i=1:length(n_dp)
[dd_4 dip_4]=n2ddda(n_dp(i,:));
dd(i,1)=dd_4;
dip(i,1)=dip_4;
end

    
dd_t=20

mag_mult=[];
mag_s=[];
dd_m=[];
M=[];
j=0;

for i=1:length (dd)-1
j=0;
    while (i+j+1<length(dd) & abs(dd(i+1+j)-dd(i))<dd_t)
        dd_mean=mean (dd(i:i+1+j));
        M(i,1)=dd_mean;
        M(i,2)=2+j;
        M(i+1+j,:)=0;
        j=j+1;
    end
    if abs(dd(i+1)-dd(i))>dd_t       
        M(i,1)=dd(i);
        M(i,2)=1; 
    end
    
end

ix=find (M(:,2)>10);

figure
%plot3 (Western10Hz, Northern10Hz, Vertical10Hz, '-k');
hold on
plot3 (yi_1, yi_2, yi_3, 'og');

for i=1:length (ix)
    plot3 (yi_1 (ix(i):ix(i)+(M(ix(i),2))), yi_2 (ix(i):ix(i)+(M(ix(i),2))), yi_3(ix(i):ix(i)+(M(ix(i),2))),'or')
    hold on;
    %, 'Linewidth',1)
end
grid on
%%
a=yi_1 (ix(i):ix(i)+(M(ix(i),2)));
b=yi_2 (ix(i):ix(i)+(M(ix(i),2)));
c=yi_2 (ix(i):ix(i)+(M(ix(i),2)));

AA=[a
    b
    c];

[ AA_dd AA_da ] = l2tp(AA(2,:),AA(1,:),AA(3,:));AA_dd=AA_dd'; AA_da=AA_da';

hold on
plot3 (yi_1 (28:32), yi_2 (28:32), yi_3(28:32),'ok')%, 'Linewidth',1)



    %%
c_sum=1:56520;
y=Western10Hz;
xi=1:50:56520;
yi=interp1(c_sum,y,xi);
plot(c_sum,y,'ok'); hold on;
plot (Time10Hz*10, Western10Hz, '-b'); hold on;
plot(xi,yi,'or'); hold on;









figure
plot(c_sum, Western10Hz(1:56519),'ok'); hold on;
plot (c_sum, Western10Hz(1:56519), '-k'); hold on;
plot(xi,yi_1,'or'); 
grid on

figure
plot(c_sum, Northern10Hz(1:56519),'ok'); hold on;
plot (c_sum, Northern10Hz(1:56519), '-k'); hold on;
plot(xi,yi_2,'ob'); 
grid on

figure
plot(c_sum, Vertical10Hz(1:56519),'ok'); hold on;
plot (c_sum, Vertical10Hz(1:56519), '-k'); hold on;
plot(xi,yi_3,'og'); 
grid on



We_vec=yi_1(2:end)-yi_1(1:end-1);
No_vec=yi_2(2:end)-yi_2(1:end-1);
Ve_vec=yi_3(2:end)-yi_3(1:end-1);

m=[We_vec,No_vec,Ve_vec];

v=[
        m(:,1)
        m(:,2)
        m(:,3)
        ];
    

%%

%%



figure
plot(Time10Hz, Pressure10Hz,'-k');
hold on
for i=1:length (ix)
    plot (Time10Hz (ix(i):ix(i)+(M(ix(i),2))), Pressure10Hz (ix(i):ix(i)+(M(ix(i),2))),'r','Linewidth',10)
end



