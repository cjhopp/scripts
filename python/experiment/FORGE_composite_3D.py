import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

import os

import matplotlib.pyplot as plt
font = {'size':16}
plt.rc('font',**font)

import utm
from scipy import interpolate



M2KM = 0.001

def load_basement_boundary(filepath, extent = [-1000,4000,-2500,2500]):
    '''
    load basin-basement boundary (sediment vs. granitoid) from reflection survey [input as utm, depth relative to zero asl]
    from https://gdr.openei.org/submissions/1107
    '''
    granitoid_bound = np.genfromtxt(filepath,delimiter=',', skip_header=1)

    ref_elev = 1650.0249
    ref_utm_e = 334641.1891
    ref_utm_n = 4263443.693

    e_gran = granitoid_bound[:,0] - ref_utm_e
    n_gran = granitoid_bound[:,1] - ref_utm_n
    z_gran = granitoid_bound[:,2] - ref_elev

    print(min(e_gran),max(e_gran),min(n_gran),max(n_gran))

    # cut granitoid boundary
    mask_x = np.logical_and(e_gran >= extent[0], e_gran <= extent[1])
    mask_y = np.logical_and(n_gran >= extent[2], n_gran <= extent[3])
    mask = np.logical_and(mask_x,mask_y)

    e_gran = e_gran[mask]
    n_gran = n_gran[mask]
    z_gran = z_gran[mask]

    return e_gran, n_gran, z_gran

def load_spac_data(spac_coords_file, spac_data_path):

    #load coordinates of depth profiles from Zhang and Pankow, 2021
    coords_spac = np.loadtxt(spac_coords_file)
    n_nodes = coords_spac.shape[0]
    print('Loading %i SPAC depth profiles' % n_nodes)

    # coordinate conversion for SPAC depth profile locations
    lon_spac = coords_spac[:,1]
    lat_spac = coords_spac[:,2]

    ### convert lat/lon to ENZ via UTM coordinates
    utm_east = np.empty((coords_spac.shape[0]))
    utm_north = np.empty((coords_spac.shape[0]))
    utm_zone = np.empty((coords_spac.shape[0]))
    utm_letter = np.empty((coords_spac.shape[0]),dtype='U1')

    for i_node in range(n_nodes):
        [utm_east[i_node], utm_north[i_node], utm_zone[i_node], utm_letter[i_node]] = utm.from_latlon(lat_spac[i_node], lon_spac[i_node])

    e_spac =  utm_east - ref_utm_e
    n_spac = utm_north - ref_utm_n
    elev_spac = coords_spac[:,3] - ref_elev
    #load SPAC velocity data for all depth profiles (depth resolution = 1m --> indice = depth)
    spac_vel_1m = np.zeros((n_nodes,max_depth+1,3))

    for i_nod in range(n_nodes):
        # python to node name correction
        i_nod_str = str(i_nod+1).zfill(2)
        data = np.genfromtxt('%s/%s_mod.lst' % (spac_data_path,i_nod_str))

        #fill SPAC velocity array, removing first line of profiles (only needed for internal SPAC procedure?)
        spac_vel_1m[i_nod,:,:] = data[1:,:]

    return e_spac, n_spac, elev_spac, spac_vel_1m


########
# DEFINE MODEL EXTENT, PARAMETERS, ETC., HERE
########

#####input files:
topography_path = 'land_surface_vertices.csv' # topography from Podgorny data publication
top_granitoid_path = 'top_granitoid_vertices.csv' # basement boundary from Podgorny
spac_coords_file = 'Vs_nodes_coord.lst' # set path to nodal coordinates file from Zhang and Pankow et al.
spac_data_path = './mod_profiles' # set path to SPAC profiles from Zhang and Pankow et al.
vpvs_path = './Vel_DAS.txt' # set path to vp-vs-profile from Lellouche et al, 2020

grid_spacing = 50          # desired grid spacing (same for all 3 directions) in m
max_depth = 3000           # max depth of model


want_control_figs = False   # output intermediate control plots
# precalc = False             # if needed for debug

# upper most velocity set to vs=400 m/s following Zhang et al, 2019, vp = vs *vp_vs_ratio
# set velocity of the upper most 30 meters to vs=400m/s (SPAC has no resolution directly at the surface, we use vs30 from Zhang et al 2019 here)
# Vs30s at FOR3: 406 +- 12 m/s (Zhang et al, 2019, FOR3 corresponds to the location of UU.FSB3 before the station was moved into a shallow borehole)
# above the surface, last layer extended until the upper end of the model (suitable for relocations)
# set to 1 if a air/vacuum layer is needed for reflections at the surface in other applications
top_velocity = 400


# replacing deep SPAC velocities with homogeneous velocities
homogen_below_basement = True
vs_ges= 3400               # HOMOGENEOUS GRANITOID ROCK VELOCITY calibrated from downhole monitoring (pers. comm. B. Dyer)
vp_ges= 5800               # HOMOGENEOUS GRANITOID ROCK VELOCITY calibrated from downhole monitoring (pers. comm. B. Dyer)
vp_vs_ratio = 1.71

#output = 'netCDF'
output = 'nll'

model_extent = 'UUSS'  # extended model including nearby UUSS permanent stations; ONLY FOR NLL
#model_extent = 'nodal' # extended model nodal deployment of Whidden et al, 2023; ONLY FOR NLL


# smoothing width over n grid points tested from 1 (no smoothing) to 10
# n gridppoints standard deviation of applied gaussian filter
# best fit to manual travel time picks at smooth_width = 5 --> 50m smoothing for 50m grid spacing
smooth_width = 5

#local reference point at Utah FORGE wellhead of 16A (injection borehole)
ref_lat = 38.50402147
ref_lon = -112.8963897
ref_elev = 1650.0249
ref_utm_e = 334641.1891
ref_utm_n = 4263443.693

# simplified downsampling of depth profile, only sample every nth data point, filter effects negligible
n_down_spac_depth = 10


#######################################
# load available velocity information #
#######################################
### Spatial Autocorrelation model (SPAC) from Zhang and Pankow, 2021
e_spac, n_spac, elev_spac, spac_vel_1m = load_spac_data(spac_coords_file, spac_data_path)
n_nodes = len(e_spac)

if want_control_figs:
    #plot nodes used in SPAC analysis on a map
    fig,axs=plt.subplots(figsize=(20,15))
    text_offset = 50
    axs.plot(e_spac,n_spac,'vb',markersize=14)

    for i_node in range(n_nodes):
        axs.text(e_spac[i_node]+text_offset,n_spac[i_node]+text_offset,i_node+1,fontsize=16)

    axs.set_xlabel('x in m')
    axs.set_ylabel('y in m')
    axs.set_aspect('equal')
    plt.tight_layout()
    fig.savefig('./map_stations.png')

####################################################
# get depth of basement boundary for each SPAC profile
####################################################

# load mesh points of basement boundary
e_gran, n_gran, z_gran = load_basement_boundary(top_granitoid_path,extent = [-1000,4000,-2500,2500])

# calculate distance between spac depth profiles and granitoid boundary mesh points
# to find boundary depth for each spac depth profile
node_coords = np.stack((e_spac,n_spac)).T
basement_coords = np.stack((e_gran,n_gran)).T
node_basement_dist_array = cdist(node_coords, basement_coords)

#get depth from reflection survey basement mesh
depth_basement = []

for i_node in range(n_nodes):
    # closest boudary mesh point for each node position
    closest_mesh = np.argsort(node_basement_dist_array[i_node])[0]
    depth_grani = round(abs(z_gran[closest_mesh]))

    # based on the top of the granitoid basement as determined from reflection profiles (GDR, )
    depth_basement.append(depth_grani)

depth_basement = np.abs(np.asarray(depth_basement, dtype=int))

print('------------------------------')
print(depth_basement)
print('------------------------------')

# add additional layers above reference elevation to account for nodes above and below reference zero depth
# allow 200 m extra on top and 100 m at bottom
add_top = 200
add_bottom = 100

vs_input = spac_vel_1m[:,:,1]
vs_input_raw = np.copy(vs_input)
#depth_array = np.arange(-add_top,max_depth+add_bottom+1,1)
depth_array_1m = np.arange(-add_top,max_depth+add_bottom+1,1)
vs = np.zeros((n_nodes,len(depth_array_1m)))
vs_raw = np.zeros((n_nodes,len(depth_array_1m)))

depth_basement += add_top

for i_node in range(n_nodes):

    #get upper and lower extent of given velocity profile depending on receiver depth/elevation
    upper_extent = int(add_top - (elev_spac[i_node])) #
    lower_extent = int(upper_extent + (max_depth+1))

    vs_input[i_node,:30] = top_velocity

    # fill and extended Vs array at top and bottom of input vs_profile
    vs[i_node,upper_extent:lower_extent] = vs_input[i_node,:]
    vs_raw[i_node,upper_extent:lower_extent] = vs_input_raw[i_node,:]

    if homogen_below_basement:
        # set homogeneous velocity below basement boundary
        vs[i_node,depth_basement[i_node]:] = vs_ges
    else:
        #extend lowest velocity to the end of the depth profile
        vs[i_node,np.argmax(vs[i_node,:]):] = np.max(vs[i_node,:])

    #same velocity in cover layer consdering differnt height of stations
    vs[i_node,:upper_extent] =  vs_input[i_node,0]
    vs_raw[i_node,:upper_extent] = vs_input_raw[i_node,0]
    #same velocity in cover layer consdering differnt height of stations
    vs[i_node,:upper_extent+30] =  top_velocity
    vs_raw[i_node,:upper_extent+30] = top_velocity

#################################################################
#CALCULATE VP by rescale vp/vs for each SPAC profile based on depth of granite boundary
#################################################################
### incorporate DAS borehole 78-32 velocity model from Lellouch et al, 2020
# ASSUMPTION1: vp/vs ratios above granite are the same everywhere but are compressed/extended if away from wells
# ASSUMPTION2: below and above ll_profile vp/vs = 1.71


### loading data
vel_ll = np.loadtxt(vpvs_path, skiprows=1)
depth_ll = vel_ll[:,0]
vp_ll = vel_ll[:,1]
vs_ll = vel_ll[:,2]

# setup vp/vs profile with constant ratio
vpvs = np.ones((vs.shape))*vp_vs_ratio
max_ll_depth = np.max(depth_ll)

for i_node in range(n_nodes):

    # normalize well profile to depth profile
    depth_ll_node = depth_ll * (depth_basement[i_node]/max_ll_depth)

    #interpolate ll onto desired depth range (basin thickness of the node depth profile)
    vp_ll_int_ = np.interp(np.arange(0,depth_basement[i_node]),depth_ll_node,vp_ll)
    vs_ll_int_ = np.interp(np.arange(0,depth_basement[i_node]),depth_ll_node,vs_ll)

    vpvs[i_node,:depth_basement[i_node]] = vp_ll_int_/vs_ll_int_

#get vp via vp/vs
vp = vs*vpvs

if want_control_figs:
    ####################################################################
    ###show profiles for checking shallow layer and depth values####
    ####################################################################
    for i_node in range(n_nodes)[::10]: # only plott every 10th profile
        fig,axs = plt.subplots(figsize=(4,20))
        axs.plot(vs[i_node,:],depth_array_1m,'k',lw=2,label='vs')
        axs.plot(vs_raw[i_node,:], depth_array_1m, lw=3, label='vs_raw', c='g', ls=':')
        axs.plot(vp[i_node,:],depth_array_1m,'k',ls='--',lw=2,label='vp')

        if homogen_below_basement:
            basement_bound = depth_array_1m[int(np.where(vs[i_node,:]>=vs_ges)[0][0])]
            axs.text(400,basement_bound, basement_bound, c='r')
            axs.axhline(y=basement_bound,ls=':',c='red',lw=2, label='vs top granitoid')

        #axs2 = axs.twiny()
        #axs2.plot(vpvs[i_node,:],depth_array_1m,'k',ls=':',lw=2,label='vp/vs')
        #axs2.set_xlabel('vp/vs ratio',color='k')

        axs.grid()
        axs.set_xlabel('velocity (m/s)')
        axs.set_title('profile no.'+str(i_node+1))
        axs.invert_yaxis()
        axs.set_ylabel('depth (m)')

        axs.plot(np.nan,np.nan,'k',ls=':',label='vp/vs')
        axs.legend(loc='lower right')
        fig.tight_layout()
        plt.show()

###########################################
# downsampling of 1m depth profiles (SPAC)
###########################################
# --> saving computational time without loosing actual information
# --> 1m input is well below the resolution of SPAC
if n_down_spac_depth > 1:
    depth_down = depth_array_1m[::n_down_spac_depth]
    vs_down = []
    vp_down = []
    for i_node in range(n_nodes):
        vs_temp = vs[i_node,:][::n_down_spac_depth]
        vp_temp = vp[i_node,:][::n_down_spac_depth]

        vs_down.append(vs_temp)
        vp_down.append(vp_temp)

    vs = np.asarray(vs_down)
    vp = np.asarray(vp_down)
    depth_array = depth_down

vs_flat = vs.flatten()
vp_flat = vp.flatten()

#construct flat grid in loop, implement numpy if too slow
e_flat = []
n_flat = []
depth_flat = []

for i_node in range(n_nodes):
    e_flat.extend([e_spac[i_node]]*len(depth_array))
    n_flat.extend([n_spac[i_node]]*len(depth_array))
    depth_flat.extend(list(depth_array))

points = np.stack((e_flat,n_flat,depth_flat)).T

###new_grid
# nodes not perfectly aligned EW NS + coordinate conversion precision
# grid adjusted to lie mostly within the area spanned by the outer most nodes
# --> more efficient and reliable,
# --> but a little offset, that is negligible consdiering the resolution of the SPAC method
e_grid_flat = np.arange(np.min(e_flat)+10,np.max(e_flat), grid_spacing)
n_grid_flat = np.arange(np.min(n_flat)+15,np.max(n_flat), grid_spacing)
z_grid_flat = np.arange(np.min(depth_flat),np.max(depth_flat)+grid_spacing, grid_spacing)

E,N,Z = np.meshgrid(e_grid_flat, n_grid_flat, z_grid_flat, indexing='ij')
out_coords = np.stack((E.flatten(),N.flatten(),Z.flatten())).T

print('Grid shape:', np.shape(E))
print('east_extent ',np.min(e_grid_flat),np.max(e_grid_flat))
print('north_extent',np.min(n_grid_flat),np.max(n_grid_flat))
print('depth_extent',np.min(z_grid_flat),np.max(z_grid_flat))

#############################################
#interpolate velocity profiles to 3D volume #
#############################################

vs_gridded =interpolate.griddata(points, vs_flat, out_coords)
vp_gridded =interpolate.griddata(points, vp_flat, out_coords)

vs_3d_direct = vs_gridded.reshape(len(e_grid_flat),len(n_grid_flat),len(z_grid_flat))
vp_3d_direct = vp_gridded.reshape(len(e_grid_flat),len(n_grid_flat),len(z_grid_flat))

vp_3d_direct = vp_3d_direct.astype(np.single)
vs_3d_direct = vs_3d_direct.astype(np.single)

np.save('vp_3d_interp.npy',vp_3d_direct)
np.save('vs_3d_interp.npy',vs_3d_direct)

########################
#smoothing model       #
########################

sigma = (smooth_width,smooth_width,smooth_width) # in gridppoints standard deviation of gaussian filter

vs_data = gaussian_filter(vs_3d_direct, sigma, mode='nearest').astype(np.single) #mode handles borders, extending using the last pixel
vp_data = gaussian_filter(vp_3d_direct, sigma, mode='nearest').astype(np.single) #mode handles borders, extending using the last pixel

vs_data_smooth = vs_data.copy()
vp_data_smooth = vp_data.copy()
###########################################################################################
#reset sharp boundary of sediment-basement interface and uppermost layers after smoothing #
###########################################################################################

# coordinates of SW grid corner (orgin of grid)
x0 = -311
y0 = -1628
z0 = -200

dx = grid_spacing

#### get topographic surface elevation ###
topo_surface = np.genfromtxt(topography_path,delimiter=',', skip_header=1)

x_topo = topo_surface[:,0] - ref_utm_e
y_topo = topo_surface[:,1] - ref_utm_n
z_topo = topo_surface[:,2] - ref_elev

#data reduction for efficiency
extent = [-1900,3800,-2300,1700]
mask_x = np.logical_and(x_topo >= extent[0], x_topo <= extent[1])
mask_y = np.logical_and(y_topo >= extent[2], y_topo <= extent[3])

mask = np.logical_and(mask_x,mask_y)

x_topo = x_topo[mask]
y_topo = y_topo[mask]
z_topo = z_topo[mask]
topo_coords = np.stack((x_topo,y_topo)).T


#### get basement boundary depth ###
data = np.genfromtxt(top_granitoid_path,delimiter=',', skip_header=1)
x_base = data[:,0] - ref_utm_e
y_base = data[:,1] - ref_utm_n
z_base = data[:,2] - ref_elev


#data reduction for efficiency
extent = [-1900,3800,-2300,1700]
mask_x = np.logical_and(x_base >= extent[0], x_base <= extent[1])
mask_y = np.logical_and(y_base >= extent[2], y_base <= extent[3])
mask = np.logical_and(mask_x,mask_y)

x_base = x_base[mask]
y_base = y_base[mask]
z_base = z_base[mask]
basement_coords = np.stack((x_base,y_base)).T

for i_vel,vel in enumerate([vs_data,vp_data]):
    print('#################')

    xs = np.asarray([x0+val*dx for val in range(np.shape(vel)[0])])
    ys = np.asarray([y0+val*dx for val in range(np.shape(vel)[1])])
    zs = np.asarray([z0+val*dx for val in range(np.shape(vel)[2])])

    # loop over model area
    for ix in range(len(xs)):
        for iy in range(len(ys)):

            # find closest mesh point of the basement boundary
            basement_coords_diff = basement_coords.copy()
            basement_coords_diff[:,0] -= xs[ix]
            basement_coords_diff[:,1] -= ys[iy]
            dists = np.sqrt(np.sum(basement_coords_diff**2,axis=1))
            basement_idx = np.argmin(dists)

            basement_depth = z_base[basement_idx]
            grid_depth_idx = int(np.round(abs((basement_depth-add_top))/(dx))) #in meters #v3_method

            # find closest mesh point of the topographic elevation
            topo_coords_diff = topo_coords.copy()
            topo_coords_diff[:,0] -= xs[ix]
            topo_coords_diff[:,1] -= ys[iy]
            dists = np.sqrt(np.sum(topo_coords_diff**2,axis=1))
            topo_idx = np.argmin(dists)

            topo_elevation = z_topo[topo_idx]
            grid_topo_idx = int(np.floor(abs(topo_elevation-add_top)/dx)) #in meters #v3_method

            if i_vel == 0:
                vel[ix,iy,grid_depth_idx:] = vs_ges
                vel[ix,iy,:grid_topo_idx] = top_velocity

            elif i_vel == 1:
                vel[ix,iy,grid_depth_idx:] = vs_ges*vp_vs_ratio
                vel[ix,iy,:grid_topo_idx] = top_velocity*vp_vs_ratio

    if i_vel == 0:
        np.save('vs_3d_interp_smoothed_%i.npy' % smooth_width ,vel)
        vs_data = vel.copy()
    elif i_vel == 1:
        np.save('vp_3d_interp_smoothed_%i.npy' % smooth_width ,vel)
        vp_data = vel.copy()


#coordinates of SW grid corner (orgin of grid)
x0 = -311
y0 = -1628
z0 = -200
dx = grid_spacing

if want_control_figs:

    fig_depmap,ax_depmap = plt.subplots()
    ####################################################################
    ###show profiles for checking shallow layer and depth values####
    ####################################################################
    xs = np.asarray([x0+val*dx for val in range(np.shape(vp_data)[0])])
    ys = np.asarray([y0+val*dx for val in range(np.shape(vp_data)[1])])
    zs = np.asarray([z0+val*dx for val in range(np.shape(vp_data)[2])])

    data = np.genfromtxt(top_granitoid_path,delimiter=',', skip_header=1)
    x_base = data[:,0] - ref_utm_e
    y_base = data[:,1] - ref_utm_n
    z_base = data[:,2] - ref_elev

    #data reduction for efficiency
    extent = [-1900,3800,-2300,1700]
    mask_x = np.logical_and(x_base >= extent[0], x_base <= extent[1])
    mask_y = np.logical_and(y_base >= extent[2], y_base <= extent[3])
    mask = np.logical_and(mask_x,mask_y)

    x_base = x_base[mask]
    y_base = y_base[mask]
    z_base = z_base[mask]

    basement_coords = np.stack((x_base,y_base)).T

    basement_diff = []

    # loop over all surface grid locations

    grid_x = []
    grid_y = []

    for ix in range(len(xs)):
        e_spac_diff = e_spac - xs[ix]
        for iy in range(len(ys)):
            grid_x.append(xs[ix])
            grid_y.append(ys[iy])

            n_spac_diff = n_spac - ys[iy]

            spac_diff = np.sqrt(e_spac_diff**2 + n_spac_diff**2)
            print('depth_basement (clostest spac)', np.argmin(spac_diff), depth_basement[np.argmin(spac_diff)])
            print('%i, %i <> %i, %i' %(e_spac[np.argmin(spac_diff)],n_spac[np.argmin(spac_diff)],xs[ix],ys[iy]))

             # find closest mesh point of the basement boundary
            basement_coords_diff = basement_coords.copy()
            basement_coords_diff[:,0] -= xs[ix]
            basement_coords_diff[:,1] -= ys[iy]
            dists = np.sqrt(np.sum(basement_coords_diff**2,axis=1))

            basement_idx = np.argmin(dists)
            basement_depth = z_base[basement_idx]
            grid_depth_idx = int(np.round(abs((basement_depth))/(dx))) #in meters

            if homogen_below_basement:
                basement_bound_smooth = zs[int(np.where(vs_data[ix,iy,:]>=vs_ges*0.9)[0][0])]
                basement_bound_unsmooth = zs[int(np.where(vs_3d_direct[ix,iy,:]>=vs_ges*0.9)[0][0])]

            fig,axs = plt.subplots(figsize=(4,20))
            axs.plot(vs_data_smooth[ix,iy,:],zs,'orange',lw=2,label='vs_smooth')
            axs.plot(vp_data_smooth[ix,iy,:],zs,'orange',ls='--',lw=2,label='vp_smooth')
            axs.plot(vs_data[ix,iy,:],zs,'k',lw=2,label='vs_smooth_cor')
            axs.plot(vs_raw[np.argmin(spac_diff),:], depth_array_1m, lw=3, label='vs_raw', c='g', ls=':')
            axs.plot(vp_data[ix,iy,:],zs,'k',ls='--',lw=2,label='vp_smooth_cor')
            axs.plot(vs_3d_direct[ix,iy,:],zs,'r',lw=2,label='vs_direct')
            axs.plot(vp_3d_direct[ix,iy,:],zs,'r',ls='--',lw=2,label='vp_direct')

            if homogen_below_basement:
                # print(zs[np.where(vs_data[ix,iy,:]>=vs_ges*0.9)[0]])
                # print(vs_data[ix,iy,:][:np.where(vs_data[ix,iy,:]>=vs_ges*0.9)[0][0]])
                # print(vs_data[ix,iy,:][np.where(vs_data[ix,iy,:]>=vs_ges*0.9)[0][0]:])

                axs.text(400,basement_bound_smooth, basement_bound_smooth, c='k')
                axs.axhline(y=basement_bound_smooth,ls=':',c='k',lw=2, label='vs top granitoid_smooth')
                # print('-------------------------')
                # print(zs[np.where(vs_3d_direct[ix,iy,:]>=vs_ges*0.9)[0]])
                # print(vs_3d_direct[ix,iy,:][:np.where(vs_3d_direct[ix,iy,:]>=vs_ges*0.9)[0][0]])
                # print(vs_3d_direct[ix,iy,:][np.where(vs_3d_direct[ix,iy,:]>=vs_ges*0.9)[0][0]:])

                axs.text(400,basement_bound_unsmooth, basement_bound_unsmooth, c='r')
                axs.axhline(y=basement_bound_unsmooth,ls=':',c='r',lw=2, label='vs top granitoid_unsmooth')



            print('basement_idx', min(dists),basement_idx, grid_depth_idx, zs[grid_depth_idx])
            print('-----------------------------')
            # axs.text(400,zs[grid_depth_idx], zs[grid_depth_idx], c='b')
            # axs.axhline(y=zs[grid_depth_idx],ls=':',c='b',lw=2, label='top granitoid_reflect')


            basement_diff.append(zs[grid_depth_idx] - basement_bound_unsmooth)

            # axs2 = axs.twiny()
            # axs2.plot(vpvs[i_node,:],depth_array,'k',ls=':',lw=2,label='vp/vs')
            # axs2.set_xlabel('vp/vs ratio',color='k')

            axs.grid()
            axs.set_xlabel('velocity (m/s)')
            #axs.set_title('profile no.'+str(i_node+1))
            axs.invert_yaxis()
            axs.set_ylabel('depth (m)')

            axs.plot(np.nan,np.nan,'k',ls=':',label='vp/vs')
            axs.legend(loc='lower right')
            fig.tight_layout()
            plt.show()

if want_control_figs:
    ##########################################
    #plot 2D slice of final interpolated model
    ##########################################

    fig,axs = plt.subplots(3,3,figsize=(20,20),sharey=True,sharex=True)

    axs[0,0].imshow(vs_data[:,:,0],
                    vmin = np.min(vs_data),
                    vmax = np.max(vs_data))

    axs[0,0].set_title('vs, y = '+str(0*grid_spacing) +' m')
    axs[0,0].set_yticks(np.linspace(0,vs_data.shape[0],5))
    axs[0,0].set_yticklabels(np.round(np.linspace(0,vs_data.shape[0],5)*grid_spacing).astype(int))
    axs[0,0].set_ylabel('z (m)')
    axs[0,0].set_aspect('equal')
    axs[0,1].imshow(vp_data[:,:,0],
                    vmin = np.min(vp_data),
                    vmax = np.max(vp_data))
    axs[0,1].set_title('vp, y = '+str(0*grid_spacing) +' m')
    axs[0,1].set_yticks(np.linspace(0,vs_data.shape[0],5))
    axs[0,1].set_yticklabels(np.round(np.linspace(0,vs_data.shape[0],5)*grid_spacing).astype(int))
    axs[0,1].set_aspect('equal')
    axs[0,2].imshow((vp_data[:,:,0]/vs_data[:,:,0]),
                        vmin = np.min(vp_data/vs_data),
                        vmax = np.max(vp_data/vs_data))
    axs[0,2].set_title('vp/vs, y = '+str(0*grid_spacing) +' m')
    axs[0,2].set_yticks(np.linspace(0,vs_data.shape[0],5))
    axs[0,2].set_yticklabels(np.round(np.linspace(0,vs_data.shape[0],5)*grid_spacing).astype(int))
    axs[0,2].set_aspect('equal')

    axs[1,0].imshow(vs_data[:,:,int(vs_data.shape[2]/2)],
                        vmin = np.min(vs_data),
                        vmax = np.max(vs_data))
    axs[1,0].set_title('vs, y = '+str(int(vs_data.shape[2]/2)*grid_spacing) +' m')
    axs[1,0].set_ylabel('z (m)')
    axs[1,0].set_aspect('equal')
    axs[1,1].imshow(vp_data[:,:,int(vs_data.shape[2]/2)],
                        vmin = np.min(vp_data),
                        vmax = np.max(vp_data))
    axs[1,1].set_title('vp, y = '+str(int(vs_data.shape[2]/2)*grid_spacing) +' m')
    axs[1,1].set_aspect('equal')
    axs[1,2].imshow((vp_data[:,:,int(vs_data.shape[2]/2)]/vs_data[:,:,int(vs_data.shape[2]/2)]),
                        vmin = np.min(vp_data/vs_data),
                        vmax = np.max(vp_data/vs_data))
    axs[1,2].set_title('vp/vs, y = '+str(int(vs_data.shape[2]/2)*grid_spacing) +' m')
    axs[1,2].set_aspect('equal')

    im=axs[2,0].imshow(vs_data[:,:,-1],
                            vmin = np.min(vs_data),
                            vmax = np.max(vs_data))
    #cax = fig.add_axes([0.1, 0, 0.2, 0.03])
    #cbar = plt.colorbar(im,cax=cax,orientation='horizontal')
    axs[2,0].set_title('vs, y = '+str(vs_data.shape[2]*grid_spacing) +' m')
    axs[2,0].set_xticks(np.linspace(0,vs_data.shape[1],5))
    axs[2,0].set_xticklabels(np.round(np.linspace(0,vs_data.shape[1],5)*grid_spacing).astype(int))
    axs[2,0].set_ylabel('z (m)')
    axs[2,0].set_xlabel('x (m)')
    axs[2,0].set_aspect('equal')
    im=axs[2,1].imshow(vp_data[:,:,-1],
                            vmin = np.min(vp_data),
                            vmax = np.max(vp_data))
    #cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    #cbar = plt.colorbar(im,cax=cax,orientation='horizontal')
    axs[2,1].set_title('vp, y = '+str(vs_data.shape[2]*grid_spacing) +' m')
    axs[2,1].set_xticks(np.linspace(0,vs_data.shape[1],5))
    axs[2,1].set_xticklabels(np.round(np.linspace(0,vs_data.shape[1],5)*grid_spacing).astype(int))
    axs[2,1].set_xlabel('x (m)')
    axs[2,1].set_aspect('equal')
    im=axs[2,2].imshow((vp_data[:,:,-1]/vs_data[:,:,-1]),
                            vmin = np.min(vp_data/vs_data),
                            vmax = np.max(vp_data/vs_data))
    #cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    #cbar = plt.colorbar(im,cax=cax,orientation='horizontal')
    axs[2,2].set_title('vp/vs, y = '+str(vs_data.shape[2]*grid_spacing) +' m')
    axs[2,2].set_xlabel('x (m)')
    axs[2,2].set_xticks(np.linspace(0,vs_data.shape[1],5))
    axs[2,2].set_xticklabels(np.round(np.linspace(0,vs_data.shape[1],5)*grid_spacing).astype(int))
    axs[2,2].set_aspect('equal')
    plt.tight_layout()
    fig.savefig('./interpolated_model_1slice_3D_int_0.png')
    plt.close(fig)



if output == 'binary':
    ##################################################
    # write to big-endian binary (e.g. for Heidimod) #
    ##################################################

    # Calculate density using Brocher (2005)
    density = ((1.6612*vp_data/1000)-(0.4721*(vp_data/1000)**2) + (0.0671*(vp_data/1000)**3) - (0.0043 *(vp_data/1000)**4) + (0.000106*(vp_data/1000)**5))*1000

    path_to_moduli = './'

    #### write complete 3D model
    nz = vp_data.shape[2]+2
    nx = vp_data.shape[0]
    ny = vp_data.shape[1]
    #calculate c11 and c44
    c11 = np.zeros(vp_data.shape,dtype='>f4')
    c44 = np.zeros(vs_data.shape,dtype='>f4')
    c11 = vp_data**2*density
    c44 = vs_data**2*density
    #fill moduli leaving 2 gp vaccuum layer at top
    moduli = np.zeros([nz,nx,ny,3],dtype='>f4')
    #reording arrays before output xyz -> zxy
    moduli[2:,:,:,0] = np.transpose(c11,(2,0,1))
    moduli[2:,:,:,1] = np.transpose(c44,(2,0,1))
    moduli[2:,:,:,2] = np.transpose(density,(2,0,1))
    moduli[0:2,:,:,2] = 0.00001
    moduli = moduli.reshape(moduli.size,order='F')
    moduli.T.tofile(path_to_moduli+'moduli_3D_dx'+str(grid_spacing)+'_nz'+str(nz)+'_nx'+str(nx)+'_ny'+str(ny))

elif output in ['NLL','nll','NonLinLoc','nonlinloc']:
    from nllgrid import NLLGrid

    for i_vel,vel_temp in enumerate([vs_data,vp_data]):

        grid_spacing_km = grid_spacing*M2KM

        vel = vel_temp.copy()

        vel *= M2KM
        vel = vel.astype('float32')

        # setup velocity grid
        grd_vel = NLLGrid()
        grd_vel.array = vel
        grd_vel.dx = grid_spacing_km
        grd_vel.dy = grid_spacing_km
        grd_vel.dz = grid_spacing_km
        grd_vel.x_orig = x0*M2KM
        grd_vel.y_orig = y0*M2KM
        grd_vel.z_orig = z0*M2KM
        grd_vel.type = 'VELOCITY'
        grd_vel.float_type = 'FLOAT'


        # setup slow_len grid
        # convert velocities to slowness and multiply by grid step(in km)... 'SLOW_LEN' is the standard format used in NLL
        grd = NLLGrid()

        slow_len = (1/vel) *grid_spacing_km
        slow_len = np.round(slow_len, 5)
        slow_len = slow_len.astype('float32')

        grd.array = slow_len #km
        grd.dx = grid_spacing_km #km
        grd.dy = grid_spacing_km #km
        grd.dz = grid_spacing_km #kmb
        grd.x_orig = x0*M2KM
        grd.y_orig = y0*M2KM
        grd.z_orig = z0*M2KM
        grd.type = 'SLOW_LEN'
        grd.float_type = 'FLOAT'

        print('MIN:', np.min(grd_vel.array), 'MAX:', np.max(grd_vel.array))

        if model_extent == 'UUSS':
            grd.nudge('south', num_layers=10)
            grd.nudge('east', num_layers=15)
            grd.nudge('west', num_layers=30)

            grd_vel.nudge('south', num_layers=10)
            grd_vel.nudge('east', num_layers=15)
            grd_vel.nudge('west', num_layers=30)
            print('EXTEND MODEL:', grd.x_orig, ' to ' , grd.x_orig+(grd.dx*np.shape(grd.array)[0]), '/',grd.y_orig,' to ', grd.y_orig+(grd.dy*np.shape(grd.array)[1]))

        elif model_extent == 'nodal':
            grd.nudge('east', num_layers=9)
            grd.nudge('south', num_layers=4)

            grd_vel.nudge('east', num_layers=9)
            grd_vel.nudge('south', num_layers=4)
            print('EXTEND MODEL:', grd.x_orig, ' to ' , grd.x_orig+(grd.dx*np.shape(grd.array)[0]), '/',grd.y_orig,' to ', grd.y_orig+(grd.dy*np.shape(grd.array)[1]))

        ###### correct velocities below the basment boundary for the extended model
        ###### the reflection seismic study providing the infromation on the basement depth covers a larger area

        vel_array = grd_vel.array
        x0_extend = grd_vel.x_orig
        y0_extend = grd_vel.y_orig
        z0_extend = grd_vel.z_orig
        dx = grd_vel.dx

        print(i_vel ,'--->', x0,y0,z0,dx)
        xs = np.asarray([x0_extend+val*dx for val in range(np.shape(vel_array)[0])])*1000
        ys = np.asarray([y0_extend+val*dx for val in range(np.shape(vel_array)[1])])*1000
        zs = np.asarray([z0_extend+val*dx for val in range(np.shape(vel_array)[2])])*1000


        if i_vel == 0:
            v_ges=vs_ges/1000
            v_top = top_velocity/1000
        else:
            v_ges=vp_ges/1000
            v_top = top_velocity/1000*vp_vs_ratio

        for ix in range(len(xs)):
            for iy in range(len(ys)):

                basement_coords_diff = basement_coords.copy()
                basement_coords_diff[:,0] -= xs[ix]
                basement_coords_diff[:,1] -= ys[iy]
                dists = np.sqrt(np.sum(basement_coords_diff**2,axis=1))

                basement_idx = np.argmin(dists)
                basement_depth = z_base[basement_idx]
                grid_depth_idx = int(np.round(abs((basement_depth-add_top))/(dx*1000))) #in meters

                vel_array[ix,iy,grid_depth_idx:] = v_ges

                # find closest mesh point of the topographic elevation
                topo_coords_diff = topo_coords.copy()
                topo_coords_diff[:,0] -= xs[ix]
                topo_coords_diff[:,1] -= ys[iy]
                dists = np.sqrt(np.sum(topo_coords_diff**2,axis=1))
                topo_idx = np.argmin(dists)

                topo_elevation = z_topo[topo_idx]
                grid_topo_idx = int(np.floor(abs(topo_elevation-add_top)/(dx*1000))) #in meters #v3_method

                vel_array[ix,iy,:grid_topo_idx] = v_top


        slow_len_array = (1/vel_array)*dx
        grd.array = slow_len_array
        grd_vel.array = vel_array


        os.makedirs('./model', exist_ok=True)
        # set basename; write to disk:

        if i_vel == 0:
            if model_extent == 'UUSS':
                grd.basename = './model/FORGE_3Dv3_%i_large.S.mod' % (smooth_width)
            elif model_extent == 'nodal':
                grd.basename = './model/FORGE_3Dv3_%i.S.mod' % (smooth_width)
            else:
                grd.basename = './model/FORGE_3Dv3_%i_spac.S.mod' % (smooth_width)

        elif i_vel == 1:
            if model_extent == 'UUSS':
                grd.basename = './model/FORGE_3Dv3_%i_large.P.mod' % (smooth_width)
            elif model_extent == 'nodal':
                grd.basename = './model/FORGE_3Dv3_%i.P.mod' % (smooth_width)
            else:
                grd.basename = './model/FORGE_3Dv3_%i_spac.P.mod' % (smooth_width)


        if i_vel == 0:
            vmin=0.0
            vmax=vs_ges*M2KM
        else:
            vmin=0.0
            vmax=vp_ges*M2KM


        if want_control_figs:
            mid1,mid2,mid3 = np.shape(grd_vel.array)
            axes2, cb2 = grd_vel.plot(slice_index=[mid1,mid2,10], cmap='viridis',vmin=0,vmax=vmax, handle=True)
            plt.show()

        # output slow_len grid
        grd.write_hdr_file()


elif output in ['netCDF', 'netcdf', 'NetCDF']:
    import xarray as xr

    # INPUT =============

    vp = np.load("vp_3d_interp.npy")
    vs = np.load("vs_3d_interp.npy")
    vp_smooth = np.load("vp_3d_interp_smoothed_6.npy")
    vs_smooth = np.load("vs_3d_interp_smoothed_6.npy")

    e = ref_utm_e + np.linspace(-32 * grid_spacing, 32 * grid_spacing, 66)
    n = ref_utm_n + np.linspace(-32 * grid_spacing, 32 * grid_spacing, 65)
    z = ref_elev + np.linspace(0, -66 * grid_spacing, 67)

    # END INPUT ==========

    units = 'm/s'
    epsg = 'EPSG:6341'
    fillval = np.nan
    description = 'Composite 3D velocity model for Utah FORGE'
    dx = grid_spacing
    dy = grid_spacing
    dz = grid_spacing
    x_orig = -311
    y_orig = -1628
    z_orig = -200
    data_type = 'velocity'

    ds = xr.Dataset({
        'vp': xr.DataArray(
            data   = vp,
            dims   = ["e", "n", "z"],
            coords = {"e": e,
                      "n": n,
                      "z": z
                      },
            attrs  = {
                '_FillValue': fillval,
                'units'     : units,
                'coordinate reference': epsg,
                'description': description,
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'x_orig': x_orig,
                'y_orig': y_orig,
                'z_orig': z_orig,
                'data_type': data_type
                }
            ),
            'vs': xr.DataArray(
            data   = vs,
            dims   = ["e", "n", "z"],
            coords = {"e": e,
                      "n": n,
                      "z": z
                      },
            attrs  = {
                '_FillValue': fillval,
                'units'     : units,
                'coordinate reference': epsg,
                'description': description,
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'x_orig': x_orig,
                'y_orig': y_orig,
                'z_orig': z_orig,
                'data_type': data_type
                }
            )
            }

        )

    ds_smooth = xr.Dataset({
        'vp_smooth': xr.DataArray(
            data   = vp_smooth,
            dims   = ["e", "n", "z"],
            coords = {"e": e,
                      "n": n,
                      "z": z
                      },
            attrs  = {
                '_FillValue': fillval,
                'units'     : units,
                'coordinate reference': epsg,
                'description': description,
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'x_orig': x_orig,
                'y_orig': y_orig,
                'z_orig': z_orig,
                'data_type': data_type
                }
            ),
        'vs_smooth': xr.DataArray(
            data   = vs_smooth,
            dims   = ["e", "n", "z"],
            coords = {"e": e,
                      "n": n,
                      "z": z
                      },
            attrs  = {
                '_FillValue': fillval,
                'units'     : units,
                'coordinate reference': epsg,
                'description': description,
                'dx': dx,
                'dy': dy,
                'dz': dz,
                'x_orig': x_orig,
                'y_orig': y_orig,
                'z_orig': z_orig,
                'data_type': data_type
                }
            )

            }

        )
    ds_smooth.to_netcdf("velocity_model_forge_smooth.nc")
    ds.to_netcdf("velocity_model_forge.nc")

    if want_control_figs:

        ds.vp.sel(n=ref_utm_n, method="nearest").T.plot()
        plt.show()

        ds_smooth.vs_smooth.sel(n=ref_utm_n, method="nearest").T.plot()
        plt.show()


if want_control_figs:
    fig_test,ax_test = plt.subplots(4,3)
    ax_flat = ax_test.flatten()
    for i_ax,e_ in enumerate([-1400,-750,-100,550,1200,1550]):
        idx_e = int(abs(-1650 - e_)/50)

        ax_flat[i_ax].imshow(vs_data[idx_e,:,:].T,extent=(-300,3200,-3500,0),vmin=0.650,vmax=3.300,cmap='viridis_r')

    for i_ax,n_ in enumerate([-100,550,1200,1850,2500,2800]):
        idx_n = int(abs(-300 - n_)/50)

        ax_flat[i_ax+6].imshow(vs_data[:,idx_n,:].T,extent=(-300,3200,-3500,0),vmin=0.650,vmax=3.300,cmap='viridis_r')

    for ax in ax_flat:
        ax.set_ylim(-2000,0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig_test.tight_layout()
    plt.show()