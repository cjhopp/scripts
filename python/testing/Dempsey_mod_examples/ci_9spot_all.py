# Set up a five spot co2 injection problem with stress
import os,sys
sys.path.append('c:\\users\\264485\\python\\pyfehm')
from fdata import*

############################################## generate grid
root0 = '9spot' 	# outer ring diagonal

cap_z = -1.5e3
cap_w = 100

x1,y1 = 4e3,4e3
z0,z1 = -2.5e3,-1e3 

zr = abs(z0-z1)

pointInjection = False
openHole = 200.
injDepth = -1.8e3

linearRelperm = True 		# toggle this to false for proper relperm numbers (much slower)
co2Dissolution = True		# toggle this to false for one less DOF

tTotal = 5.*365.25 	# 5 year injection operation

# sensitivity - first is default, rest to be tested
spacing = [200.,100.,150.,250.,300.,350.] 						# spacing in nine spot pattern
tStop = np.array([1.5,0.5,1.,2.,2.5,3.,3.5,4.,4.5,5.])*365.25 	# time to shift pumping to outer ring
dP = [1.5,0.5,1.,2.,2.5,3.,3.5,4.] 								# underpressure for wells

# assemble parameters and root names
roots = [root0+'_bnchmrk_0']
pars = [[spacing[0],tStop[0],dP[0]],]
cnt = 1
for sp in spacing[1:]:
	roots.append(root0+'_dx'+str(int(sp))+'m_'+str(cnt))
	pars.append([sp,tStop[0],dP[0]])
	cnt+=1
for tS in tStop[1:]:
	roots.append(root0+'_dt'+str(int(tS*12/365.25))+'months_'+str(cnt))
	pars.append([spacing[0],tS,dP[0]])
	cnt+=1
for dp in dP[1:]:
	roots.append(root0+'_dp'+str(int(dp*10))+'bar_'+str(cnt))
	pars.append([spacing[0],tStop[0],dp])
	cnt+=1
	
for root in roots: print root
for par in pars: print par

hprop = [0.325,.35,0.325]
hdiv = [4,28,4]
#hdiv = [3,10,3]

zprop = [abs(z0-(cap_z-4*cap_w))/zr, 4.5*cap_w/zr,abs(z1-(cap_z+0.5*cap_w))/zr]

zdiv = [5,20,5]
#zdiv = [4,10,4]

first = True
startAt = 1

# CJH Looping over sets of parameters on same grid
for root,par in zip(roots[startAt:],pars[startAt:]):
	# only need to make the grid once
	if first:
		gm = gmake(meshname='ci_'+root0+'_GRID.inp',xmin=0,xmax=x1,ymin=0,ymax=y1,zmin=z0,zmax=z1,
			xprop=hprop, yprop=hprop, xdiv=hdiv, ydiv=hdiv, zprop=zprop, zdiv=zdiv)
		gm.run()
		
	dat = fdata()
	dat.grid.read('ci_'+root0+'_GRID.inp')
	if first:
		dat.grid.plot(color='b',cutaway='middle',save = 'ci_'+root0+'_GRID.png')
	
	############################################## assign rock units
	dat.add(fmacro('perm',zone=0,param=(('kx',1.e-16),('ky',1.e-16),('kz',1.e-16))))
	dat.add(fmacro('rock',zone=0,param=(('density',2500),('porosity',0.01),('specific_heat',900))))
	dat.add(fmacro('cond',zone=0,param=(('cond_x',1.5),('cond_y',1.5),('cond_z',1.5))))
	
	############################################## assign cap rock
	x0,y0 = dat.grid.xmin, dat.grid.ymin
	cz = fzone(index=10,name='caprock')
	cz.rect([x0-0.01,y0-0.01,cap_z-0.5*cap_w-0.01],[x1+0.01,y1+0.01,cap_z+0.5*cap_w+0.01])
	dat.add(cz)
	dat.add(fmacro('perm',zone='caprock',param=(('kx',1.e-17),('ky',1.e-17),('kz',1.e-18))))
	dat.add(fmacro('rock',zone='caprock',param=(('density',2500),('porosity',0.005),('specific_heat',900))))
	dat.add(fmacro('cond',zone='caprock',param=(('cond_x',1.5),('cond_y',1.5),('cond_z',1.5))))
	
	############################################## assign reservoir rock
	x0,y0 = dat.grid.xmin, dat.grid.ymin
	cz = fzone(index=20,name='reservoir')
	cz.rect([x0-0.01,y0-0.01,-2.2e3],[x1+0.01,y1+0.01,cap_z-0.5*cap_w-0.01])
	dat.add(cz)
	dat.add(fmacro('perm',zone='reservoir',param=(('kx',1.e-14),('ky',1.e-14),('kz',1.e-15))))
	dat.add(fmacro('rock',zone='reservoir',param=(('density',2500),('porosity',0.1),('specific_heat',900))))
	dat.add(fmacro('cond',zone='reservoir',param=(('cond_x',1.5),('cond_y',1.5),('cond_z',1.5))))
	
	############################################## assign initial conditions
	dat.add(fmacro('grad',zone=0,param=(('reference_coord',dat.grid.zmax),('direction',3),
	('variable',1),('reference_value',abs(dat.grid.zmax)*9.81*1e3/1e6),('gradient',-9.81*1e3/1e6))))
	dat.add(fmacro('grad',zone=0,param=(('reference_coord',dat.grid.zmax),('direction',3),
	('variable',2),('reference_value',25.),('gradient',-0.03))))
	dat.nobr = True
	
	############################################## assign output
	dat.cont.variables.append(['temperature','pressure','co2s','co2m','liquid','xyz','permeability'])
	dat.cont.timestep_interval = 1000
	dat.cont.file_tag = 'day'
	
	############################################## run model
	dat.time['max_timestep_NSTEP'] = 1
	
	dat.ctrl['min_timestep_DAYMIN'] = 1.e-10
	dat.ctrl['max_timestep_DAYMAX'] = 10000.
	dat.ctrl['max_newton_iterations_MAXIT'] = 40
	dat.ctrl['max_multiply_iterations_IAMM'] = 40
	dat.ctrl['stor_file_LDA']=0
	dat.ctrl['number_orthogonalizations_NORTH'] = 200
	dat.ctrl['max_solver_iterations_MAXSOLVE'] = 200
	dat.ctrl['order_gauss_elim_NAR'] = 4
	
	dat.iter['stop_criteria_NRmult_G3']=0.01
	dat.iter['machine_tolerance_TMCH']=-0.01
	
	if first:
		dat.files.root = 'ci_'+root
		dat.files.rsto = 'ci_'+root0+'_spinup_INCON.ini'
		dat.run(input='ci_'+root0+'_INPUT.dat',files=['check','hist','outp'],
                exec_path='c:\\users\\264485\\fehm\\raj_co2restart\\FEHM_open_v022813_v2.exe')
		first = False
	dat.incon.read('ci_'+root0+'_spinup_INCON.ini')
	
	############################################## phase 1, inner ring
	if co2Dissolution:
		dat.carb.on(iprtype=4)
	else:
		dat.carb.on(iprtype=3)
		
	injNode = dat.grid.node_nearest_point([x1/2,y1/2,injDepth])
	if pointInjection:
		dat.add(fzone(index=1,name='co2_injector',type='nnum',nodelist =[injNode]))
	else:
		pt = np.array(injNode.position)
		injZone = fzone(index=1,name='co2_injector')
		injZone.rect(pt-np.array([0.01,0.01,0.01]),pt + np.array([0.01,0.01,openHole+.01]))
		dat.add(injZone)
	dat.add(fmacro('co2flow',zone=1,param=(('rate',0),('energy',-40),('impedance',100),('bc_flag',1))))
	
	spacing = par[0]
	tStop = par[1]
	dP = -par[2]
	MP = []
	
	# assign producers
	zs = np.unique([nd.position[2] for nd in dat.grid.nodelist])
	zinds = np.where((zs>(pt[2]-0.01))*(zs<(pt[2]+openHole+0.01)))[0]
	zs = [zs[i] for i in zinds]
	zind = 2
	for z in zs:
		# inner corners
		proNode1 = dat.grid.node_nearest_point([pt[0]-spacing,pt[1]-spacing,z])
		proNode2 = dat.grid.node_nearest_point([pt[0]-spacing,pt[1]+spacing,z])
		proNode3 = dat.grid.node_nearest_point([pt[0]+spacing,pt[1]-spacing,z])
		proNode4 = dat.grid.node_nearest_point([pt[0]+spacing,pt[1]+spacing,z])
		dat.hist.nodelist.append(proNode1);dat.hist.nodelist.append(proNode2)
		dat.hist.nodelist.append(proNode3);dat.hist.nodelist.append(proNode4)
		# outer corners
		proNode5 = dat.grid.node_nearest_point([pt[0]-2*spacing,pt[1]-2*spacing,z])
		proNode6 = dat.grid.node_nearest_point([pt[0]-2*spacing,pt[1]+2*spacing,z])
		proNode7 = dat.grid.node_nearest_point([pt[0]+2*spacing,pt[1]-2*spacing,z])
		proNode8 = dat.grid.node_nearest_point([pt[0]+2*spacing,pt[1]+2*spacing,z])
		dat.hist.nodelist.append(proNode5);dat.hist.nodelist.append(proNode6)
		dat.hist.nodelist.append(proNode7);dat.hist.nodelist.append(proNode8)
		
		meanPressure = np.mean([
		proNode1.variable['P'],proNode2.variable['P'],proNode3.variable['P'],proNode4.variable['P'],
		proNode5.variable['P'],proNode6.variable['P'],proNode7.variable['P'],proNode8.variable['P'],
		])
		MP.append(meanPressure)
		dat.add(fzone(index=zind,type='nnum',nodelist =[proNode1,proNode2,proNode3,proNode4]))
		dat.add(fzone(index=10+zind,type='nnum',nodelist =[proNode5,proNode6,proNode7,proNode8]))
		dat.add(fmacro('flow',zone=zind,param=(('rate',meanPressure+dP),('energy',-40),('impedance',1.e-6))))
		dat.add(fmacro('co2flow',zone=zind,param=(('rate',meanPressure+dP),('energy',-40),('impedance',1.e-6),('bc_flag',3))))
		zind +=1
	
	dat.add(fmacro('co2pres',param=(('pressure',0.),('temperature',25.),('phase',1))))
	dat.add(fmacro('co2frac',param=(('water_rich_sat',1.),('co2_rich_sat',0.),('co2_mass_frac',0.),
	('init_salt_conc',0.),('override_flag',0))))
	
	if linearRelperm:
		rlp = fmacro('rlp',index=17,param=[.05,1,1,0,1,1,0,0,1,1,1,0,1,0])
	else:
		rlp = fmacro('rlp',index=17,param=[.05,1,1,0,1,1,0,0,1,1,1,0,1,0])
	dat.add(rlp)
	
	dat.time['initial_year_YEAR']=0.
	dat.time['initial_month_MONTH']=0.
	dat.time['initial_day_INITTIME']=0.
	dat.ctrl['max_timestep_DAYMAX'] = 365.*1
	dat.time['max_timestep_NSTEP'] = 1000.
	dat.time['max_time_TIMS'] = tStop
	
	dat.hist.nodelist.append(injNode)
	dat.hist.variables.append(['flow'])
	dat.hist.type = 'tec'
	
	############################################# fix side boundary conditions to allow outflow
	x0,x1 = dat.grid.xmin,dat.grid.xmax
	y0,y1 = dat.grid.ymin,dat.grid.ymax
	z0,z1 = dat.grid.zmin,dat.grid.zmax
	zn = fzone(index=35,name='xmin'); zn.rect([x0-0.01,y0-0.01,z0-0.01],[x0+0.01,y1+0.01,z1+0.01]); dat.add(zn)
	zn = fzone(index=36,name='xmax'); zn.rect([x1-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]); dat.add(zn)
	zn = fzone(index=37,name='ymin'); zn.rect([x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y0+0.01,z1+0.01]); dat.add(zn)
	zn = fzone(index=38,name='ymax'); zn.rect([x0-0.01,y1-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]); dat.add(zn)
	for zn in [35,36,37,38]:
		dat.add(fmacro('flow',zone=zn,param=(('rate',0),('energy',-30.),('impedance',100.))))
	#	dat.add(fmacro('co2flow',zone=zn,param=(('rate',0),('energy',-30.),('impedance',100.),('bc_flag',3))))
	
	os.system('del ci_'+root+'*.csv')
	
	dat.files.rsto = 'ci_'+root+'_INCON.ini'
	dat.run(input='ci_'+root+'_phase1_INPUT.dat',files=['check','hist','outp'],exec_path='c:\\users\\264485\\fehm\\raj_co2restart\\FEHM_open_v022813_v2.exe')
	dat.incon.read('ci_'+root+'_INCON.ini')
	
	os.system('copy ci_'+root+'.outp ci_'+root+'Phase1.outp')
	os.system('copy ci_'+root+'_flow_his.dat ci_'+root+'_Phase1_flow_his.dat')
	
	############################################# phase 2, outer ring
	for i,mp in enumerate(MP):
		dat.delete(dat.flow[i+2])
		dat.co2flow[i+2].param['rate']=mp
		dat.add(fmacro('flow',zone=i+2+10,param=(('rate',mp+dP),('energy',-40),('impedance',1.e-6))))
		dat.add(fmacro('co2flow',zone=i+2+10,param=(('rate',mp+dP),('energy',-40),('impedance',1.e-6),('bc_flag',3))))
		
	dat.time['initial_year_YEAR']=None
	dat.time['initial_month_MONTH']=None
	dat.time['initial_day_INITTIME']=None
	
	dat.time['max_time_TIMS'] = tTotal
	
	dat.run(input='ci_'+root+'_phase2_INPUT.dat',files=['check','hist','outp'],exec_path='c:\\users\\264485\\fehm\\raj_co2restart\\FEHM_open_v022813_v2.exe')
	
	os.system('copy ci_'+root+'.outp ci_'+root+'Phase2.outp')
	os.system('copy ci_'+root+'_flow_his.dat ci_'+root+'_Phase2_flow_his.dat')
############################################### visualise output
#execfile('ci_fivespot_plot.py')

















