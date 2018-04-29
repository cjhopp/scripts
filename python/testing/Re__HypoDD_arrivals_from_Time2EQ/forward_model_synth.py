# python script to perform synthetic locations from relocated events, and
# combine with previously known polarity data
import subprocess, glob, os

# read in list of EQ times and hypoDD relocations
# needs to be a text file with 4
# columns of date, lat, lon, depth
infile = 'gisfm_SYNTH_best_locations.csv'
NLL_cont_file_in='run/NLL_control_NZ3D_SYNTH.in.h'
NLL_cont_file_out='run/NLL_control_NZ3D_SYNTH.out.h'

EQ_times, best_lats, best_lons, best_depths = [], [], [], []

with open(infile, 'r') as f:
    for line in f:
        EQ_times.append(str(line.split(' ')[0]))
        best_lats.append(str(line.split(' ')[1]))
        best_lons.append(str(line.split(' ')[2]))
        best_depths.append(str(line.split(' ')[3].split('\n')[0]))
for i, EQ in enumerate(EQ_times):
    with open(NLL_cont_file_in) as f, open(NLL_cont_file_out, 'w') as fout:
        for m, line in enumerate(f):
            # N.B. CHECK THIS NUMBER FOR YOUR FILE
            if m == 141:
                line = "EQSRCE EV0 LATLON {0} {1} {2}\n".format(
                    best_lats[i], best_lons[i], best_depths[i])
            fout.write(line)
    subprocess.call(["Time2EQ", NLL_cont_file_out])
    subprocess.call(["NLLoc", NLL_cont_file_out])
    out_file_hyp = glob.glob('loc/*Synth.1900*.hyp')
    old_file = 'loc/NZ3D_Gau_GISFM.{0}.grid0.loc.hyp'.format(EQ)
    picked_stations=[]
    picked_polarities=[]
    with open(old_file) as f:
        for j, line in enumerate(f):
            if j > 15 and not line.startswith('END') and not line.startswith('\n'):
                if (str(line.rstrip()[26:27]) == 'D'
                    or str(line.rstrip()[26:27]) in ['C', 'U']):
                    picked_stations.append(str(line.rstrip()[0:6]))
                    picked_polarities.append(str(line.rstrip()[26:27]))
    # Get the original pick lines
    with open(out_file_hyp[0], 'r') as f:
        lines = f.read().split("\n")
    # Get the correct header info
    with open(out_file_hyp[0], 'r') as f:
        header = []
        for line in f:
            header.append(line)
            if line.startswith("PHASE"):  # Last line of header before picks
                break
    # Write it all out
    with open("loc/tmp.hyp", "w") as fout:
        #for line in header:
            #fout.write(line)
        for line in lines:
            for k, station in enumerate(picked_stations):
                if line.startswith(station) and line[19:20] == 'P':
                    line_new = line[:26] + picked_polarities[k] + line[27:]
                    break
                else:
                    line_new=line
            fout.write(line_new + "\n")
    # Note: Uncomment this if you are happy with the contents of tmp.hyp!
    #os.rename("loc/tmp.hyp", 'loc/NZ3D_Gau_Synth.{0}.grid0.loc.hyp'.format(EQ))
    # Cleanup
    for f in glob.glob('loc/*Synth.1900*.scat'):
       os.rename(f, 'loc/NZ3D_Gau_Synth.{0}.grid0.loc.scat'.format(EQ))
    for f in glob.glob('loc/*Synth.1900*.hdr'):
       os.rename(f, 'loc/NZ3D_Gau_Synth.{0}.grid0.loc.hdr'.format(EQ))
    for fil in glob.glob('loc/*Synth.1900*'):
       os.remove(fil)
      
