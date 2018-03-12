
from fdata import*

# thm1 = just try get things working

def model_setup():
	"""
	Create mesh and initialize PyFEHM object with zones.
	"""
	root = 'dp'
	# exe = r'C:\Users\264485\python\desert_peak\full_model_v2\fehm_DED_29Jan14.exe'
	dat = fdata(work_dir = 'mesh')
	dat.files.root = root

	initialConditions = True

	# grid generation - 1/8 symmetry
	#  - z = vertical, well, Sv
	#  - y = parallel to fracture, SHmax
	#  - x = perpendicular to fracture, Shmin

	# variable parameters
	dx = 3.

	# three zones of stimulation
	depth = 1035.-115. 				# depth of injection point
	dim = 2000.					# dimensions of box

	# find pair of zones which are closest
	xmin,xmid,xmax = 0., 45., dim		# dimensions of domain (high res region)
	Nx, xPower = 6, 1.5

	x = (list(np.linspace(xmin,xmid,xmid/dx+1))+
		 list(powspace(xmid,xmax,Nx+1,xPower))[1:])

	z = (list(np.linspace(-depth,-depth+xmid,xmid/dx+1))+
		 list(powspace(-depth+xmid,0,Nx+1,xPower))[1:])

	dat.grid.make(root+'_GRID.inp',x=x,y=x,z=z)

	dat.paraview()

	dat.work_dir = 'steady'

	print 'grid contains '+str(len(x)**3)+' nodes'

	# define zones
	dat.new_zone(1,'damage',rect=[[0,0,-depth],[xmid-0.1,xmid-0.1,xmid-0.1-depth]])

	injNode = dat.grid.node_nearest_point([0,0,-depth])
	dat.new_zone(2,'injection',nodelist = injNode)

	print 'damage zone contains '+str(len(dat.zone['damage'].nodelist))+' nodes'

	# material properties
	#kx0 = 1.e-16*3.	# permeability
	#ky0 = 5.e-16*3.	# permeability
	#kz0 = 1.e-16*3.	# permeability
	#dat.zone[0].permeability = [kx0,ky0,kz0]
	dat.zone[0].permeability = 1.e-15
	dat.zone[0].porosity = 0.1
	rho = 2477.
	dat.zone[0].density = rho
	dat.zone[0].specific_heat = 800.
	dat.zone[0].conductivity = 2.7

	# initial temperature/pressure conditions
	dat.temperature_gradient('dp_modelTemperatures.txt', hydrostatic = 0.1, offset=0.,first_zone = 600,auxiliary_file = 'temp.macro')
	dat.zone['ZMAX'].fix_pressure(0.1)

	# initial run allowing equilibration
	dat.files.rsto = root+'_INCON.ini'
	dat.tf = 365.25
	dat.dtmax = dat.tf

	dat.cont.variables.append(['xyz','pressure','temperature','stress','permeability'])

	if initialConditions:
		dat.run(root +'_INPUT.dat',exe=exe,use_paths = True)
	dat.incon.read(dat.work_dir + os.sep+root+'_INCON.ini')
	dat.ti = 0.

	dat.delete(dat.preslist)
	dat.delete([zn for zn in dat.zonelist if (zn.index >= 600 and zn.index <= 700)])

	# stress parameters
	mu = 0.75

	xgrad = 0.61
	ygrad = (0.61+1)/2

	dat.strs.on()
	dat.strs.bodyforce = False

	dat.add(fmacro('stressboun',zone='XMIN',param=(('direction',1),('value',0))))
	dat.add(fmacro('stressboun',zone='YMIN',param=(('direction',2),('value',0))))
	dat.add(fmacro('stressboun',zone='ZMIN',param=(('direction',3),('value',0))))
	dat.add(fmacro('stressboun',zone='XMAX',param=(('direction',1),('value',0))))
	dat.add(fmacro('stressboun',zone='YMAX',param=(('direction',2),('value',0))))

	dat.zone[0].youngs_modulus = 25.e3
	dat.zone[0].poissons_ratio = 0.2
	dat.zone[0].thermal_expansion = 3.5e-5
	dat.zone[0].pressure_coupling = 1.

	dat.incon.stressgrad(xgrad = xgrad, ygrad = ygrad, zgrad=115.*rho*9.81/1e6, calculate_vertical = True, vertical_fraction = True)
	dat.incon.write(root+'_INCON.ini')
	dat.incon.read(dat.work_dir+os.sep+root+'_INCON.ini')

	nd = dat.grid.node_nearest_point([0,0,-(930-115)])

	dat.work_dir = None

	dat.picklable()

	return dat
	
def model_run(pars, dat):
	injTime = 66.5				# length of injection in days
	injNode = dat.zone['injection'].nodelist[0]
	injNode = dat.grid.node[injNode]

	# stress parameters
	stress_proximity = 2. 			# distance of horizontal stress from critical value
	mu = 0.75

	cohesion = pars['cohesion']
	dk = pars['dk']
	exe = r'C:\Users\264485\python\desert_peak\full_model_v2\fehm_DED_29Jan14.exe'

	dat.delete(dat.permmodellist)
	pm = fmodel('permmodel',index=25,zonelist = 'damage')
	pm.param['shear_frac_tough'] = 500.
	pm.param['static_frict_coef'] = mu
	pm.param['dynamic_frict_coef'] = 0.65
	pm.param['frac_num'] = 100
	pm.param['onset_disp'] = 1.e-3
	pm.param['disp_interval'] = 6.e-3
	pm.param['max_perm_change'] = dk
	pm.param['frac_cohesion'] = cohesion
	dat.add(pm)

	# stress stuff

	# injection source
	# run multiple simulations

	data = np.loadtxt(r'C:\Users\264485\python\desert_peak\full_model_v2\dp_bottomhole_conditions.dat',skiprows=1)
	ti = data[:,0]
	Pi = list(data[:,1])
	Ti = list(data[:,5])
	P0 = injNode.Pi

	for i in range(len(ti)):
		if Pi[i] == 0.: Pi[i] = 1000.
		else: Pi[i] += P0

	dat.time['max_time_TIMS'] =66.5
	dat.add(fboun(type='ti_linear',zone=['injection'],times=ti,variable=[['ft']+Ti,['pw']+Pi]))

	# run simulation
	dat.ti = 0
	dat.tf = injTime
	dat.dtn = 5000
	#dat.dtn = 1
	dat.dtmax = 0.1

	dat.cont.variables = ['xyz','temperature','pressure','permeability','stress']
	#	if injWHP == 3. and injTemp == 100. and biot == 1. and alpha == 3.5e-5:
	dat.cont.time_interval = 0.5
	#	else:
	#	dat.cont.time_interval = 1.e30

	dat.hist.nodelist.append(injNode)
	dat.hist.variables.append(['flow'])
	dat.hist.time_interval = 0.1

	shutil.copy(
			r'C:\Users\264485\python\desert_peak\full_model_v2\fracture_orientations.dat', 				# source
			'fracture_orientations.dat'			# destination, note: file MUST be called fracture_orientations.dat
			)

	dat.files.rsto = ''

	dat.run(dat.files.root +'_INPUT.dat',exe=exe, use_paths = True, verbose=False)


	obsnames = ['null']
	outdict = dict(zip(obsnames,[0.]))

	return outdict














