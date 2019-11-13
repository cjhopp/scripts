
# =============================================================================
#  Samba NonLinLoc programs control file
#
#  NonLinLoc Version 2.3 - APR2001
# =============================================================================


# non-nested include files allowed, use:
# INCLUDE <include_file_name>


# =============================================================================
# =============================================================================
# Generic control file statements
# =============================================================================
#

CONTROL 1 54321

TRANS SIMPLE -41.75 173.0 140

# =============================================================================
# END of Generic control file statements
# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================
# Vel2Grid control file statements
# =============================================================================
#

VGOUT  ./model/vel

VGTYPE S

VGGRID  600 600 400  0.0 0.0 -3.0  0.5 0.5 0.5  SLOW_LEN

# -----------------------------------------------------------------------------

#
#
# =============================================================================
# END of Vel2Grid control file statements
# =============================================================================
# =============================================================================




# =============================================================================
# =============================================================================
# Grid2Time control file statements
# =============================================================================
#

GTFILES ./model/NZ3D ./time/NZ3D S 0

GTMODE GRID3D ANGLES_YES

GTSRCE LBPR1 LATLONDM 38 56.11 S 178 10.22 E 0.0 -0.078
GTSRCE LBPR2 LATLONDM 38 52.00 S 178 12.42 E 0.0 -0.079
GTSRCE LBPR3 LATLONDM 38 48.31 S 178 16.97 E 0.0 -0.073
GTSRCE LBPR4 LATLONDM 38 40.92 S 178 22.60 E 0.0 -0.079
GTSRCE LBPR5 LATLONDM 38 34.53 S 178 25.08 E 0.0 -0.076
GTSRCE LOB10 LATLONDM 39 8.00 S 178 18.79 E 0.0 -1.444
GTSRCE LOBS1 LATLONDM 38 35.53 S 178 49.12 E 0.0 -0.994
GTSRCE LOBS2 LATLONDM 38 37.26 S 179 2.77 E 0.0 -3.562
GTSRCE LOBS3 LATLONDM 38 47.53 S 179 8.84 E 0.0 -3.540
GTSRCE LOBS4 LATLONDM 39 7.21 S 178 58.89 E 0.0 -3.440
GTSRCE LOBS5 LATLONDM 39 7.16 S 178 41.90 E 0.0 -2.361
GTSRCE LOBS6 LATLONDM 38 58.67 S 178 47.76 E 0.0 -1.874
GTSRCE LOBS7 LATLONDM 38 42.69 S 178 34.09 E 0.0 -0.784
GTSRCE LOBS8 LATLONDM 38 50.59 S 178 27.56 E 0.0 -0.651
GTSRCE LOBS9 LATLONDM 39 4.30 S 178 31.28 E 0.0 -1.457
GTSRCE TXBP1 LATLONDM 38 45.38 S 178 59.84 E 0.0 -3.538
GTSRCE TXBP2 LATLONDM 38 42.81 S 178 34.12 E 0.0 -0.779
GTSRCE TXBP3 LATLONDM 38 50.51 S 178 40.27 E 0.0 -1.061
GTSRCE TXBP4 LATLONDM 39 0.49 S 178 28.64 E 0.0 -1.412
GTSRCE TXBP5 LATLONDM 38 56.87 S 178 34.33 E 0.0 -1.246
GTSRCE BKZ LATLONDM 39 9.94 S 176 29.55 E 0.0 0.706
GTSRCE HAZ LATLONDM 37 45.37 S 177 46.96 E  0.0 0.081
GTSRCE URZ LATLONDM 38 15.55 S 177 6.65 E  0.0 0.095
GTSRCE ARHZ LATLONDM 39 15.79 S 176 59.75 E 0.0 0.270
GTSRCE CNGZ LATLONDM 38 29.02 S 178 12.45 E 0.0 0.179
GTSRCE CKHZ LATLONDM 39 39.49 S 177 04.66 E 0.0 0.209
GTSRCE KNZ LATLONDM 39 1.31 S 177 40.42 E  0.0 0.032
GTSRCE MWZ LATLONDM 38 20.04 S 177 31.67 E 0.0 0.573
GTSRCE MXZ LATLONDM 37 33.74 S 178 18.40 E 0.0 0.106
GTSRCE PUZ LATLONDM 38 4.29 S 178 15.43 E 0.0 0.430
GTSRCE RTZ LATLONDM 38 36.92 S 176 58.83 E 0.0 0.867
GTSRCE MHGZ LATLONDM 39 9.16 S 177 54.42 E 0.0 0.302
GTSRCE MTHZ LATLONDM 38 51.14 S 176 50.47 E 0.0 0.620
GTSRCE NMHZ LATLONDM 39 5.83 S 176 48.39 E 0.0 0.864
GTSRCE PKGZ LATLONDM 37 51.98 S 178 4.83 E 0.0 0.405
GTSRCE PRGZ LATLONDM 38 55.36 S 177 53.00 E 0.0 0.503
GTSRCE RAGZ LATLONDM 38 29.74 S 177 24.92 E 0.0 0.947
GTSRCE RAHZ LATLONDM 38 54.97 S 177 5.16 E 0.0 0.472
GTSRCE RIGZ LATLONDM 38 42.35 S 177 45.73 E 0.0 0.621
GTSRCE RUGZ LATLONDM 37 57.91 S 177 40.65 E 0.0 1.175
GTSRCE SNGZ LATLONDM 38 46.89 S 177 20.39 E 0.0 0.545
GTSRCE TKGZ LATLONDM 38 26.35 S 177 50.91 E 0.0 0.157
GTSRCE TWGZ LATLONDM 38 10.60 S 177 58.82 E 0.0 0.687
GTSRCE WHHZ LATLONDM 39 4.61 S 177 14.06 E 0.0 0.317
GTSRCE WMGZ LATLONDM 37 49.15 S 178 24.89 E 0.0 0.190
GTSRCE EBS1 LATLONDM 38 44.75 S 178 40.73 E 0.0 -0.995
GTSRCE EBS2 LATLONDM 38 46.63 S 178 35.01 E 0.0 -0.930
GTSRCE EBS3 LATLONDM 38 41.68 S 178 39.04 E 0.0 -1.023
GTSRCE EBS4 LATLONDM 38 41.33 S 178 49.19 E 0.0 -1.712
GTSRCE EBS5 LATLONDM 38 59.66 S 178 19.54 E 0.0 -1.348


GT_PLFD  1.0e-3  0

#
#
# =============================================================================
# END of Grid2Time control file statements
# =============================================================================
# =============================================================================





# =============================================================================
# =============================================================================
# Time2EQ control file statements
# =============================================================================
#
#

EQFILES ./time/NZ3D ./obs/synth_Raukumara_cluster.obs

EQMODE SRCE_TO_STA

#EQEVENT  EQ001   50.0 50.0 10.0  0.0

#EQEVENT EV2 -26.7852 -7.70703 65.0332 0




#EQSRCE EV00 LATLON -38.202531 178.314876 18.623 0
#EQSRCE EV01 LATLON -38.195671 178.34318 18.724 0
#EQSRCE EV02 LATLON -38.199194 178.337467 17.758 0
#EQSRCE EV03 LATLON -38.466089 177.973063 22.131 0
#EQSRCE EV04 LATLON -38.46366 177.971663 22.075 0
#EQSRCE EV05 LATLON -38.19893 178.187907 22.543 0
#EQSRCE EV06 LATLON -38.2415 178.090169 25.593 0
#EQSRCE EV07 LATLON -38.601213 177.858464 17.71 0
#EQSRCE EV08 LATLON -38.173145 178.268359 16.234 0
#EQSRCE EV09 LATLON -38.170805 178.273063 17.096 0
#EQSRCE EV10 LATLON -38.198295 178.122656 15.422 0
#EQSRCE EV11 LATLON -38.19679 178.123014 15.41 0
#EQSRCE EV12 LATLON -38.197172 178.121077 15.634 0
#EQSRCE EV13 LATLON -38.180725 178.182731 21.712 0
#EQSRCE EV14 LATLON -38.37358 178.144189 17.198 0
#EQSRCE EV15 LATLON -38.640662 177.817269 19.57 0
#EQSRCE EV16 LATLON -38.17203 178.185465 24.694 0
#EQSRCE EV17 LATLON -38.424434 177.889274 20.243 0
#EQSRCE EV18 LATLON -38.523669 177.962923 19.979 0
#EQSRCE EV19 LATLON -38.178959 178.193522 19.582 0
#EQSRCE EV20 LATLON -38.183358 178.19585 22.125 0
#EQSRCE EV21 LATLON -38.183 178.196224 22.199 0
#EQSRCE EV22 LATLON -38.538509 177.915186 20.458 0
#EQSRCE EV23 LATLON -38.53868 177.915055 20.176 0
#EQSRCE EV24 LATLON -38.17867 178.198926 22.398 0
#EQSRCE EV25 LATLON -38.086804 178.298128 23.737 0
#EQSRCE EV26 LATLON -38.195597 178.290332 18.034 0
#EQSRCE EV27 LATLON -38.194104 178.295312 18.213 0
#EQSRCE EV28 LATLON -38.200553 178.29375 17.513 0
#EQSRCE EV29 LATLON -38.200716 178.28418 17.369 0
#EQSRCE EV30 LATLON -38.196899 178.284945 17.582 0
#EQSRCE EV31 LATLON -38.208964 178.325439 17.633 0
#EQSRCE EV32 LATLON -38.20883 178.317627 17.235 0
#EQSRCE EV33 LATLON -38.193441 178.288835 18.345 0
#EQSRCE EV34 LATLON -38.19906 178.291764 17.462 0
#EQSRCE EV35 LATLON -38.198234 178.294027 17.482 0
#EQSRCE EV36 LATLON -38.454028 178.007536 17.503 0
#EQSRCE EV37 LATLON -38.522526 177.876156 20.673 0
#EQSRCE EV38 LATLON -38.151562 178.418359 15.04 0
#EQSRCE EV39 LATLON -38.099524 178.249056 16.065 0
#EQSRCE EV40 LATLON -38.19169 178.323926 16.703 0
#EQSRCE EV41 LATLON -38.190894 178.330843 16.481 0
#EQSRCE EV42 LATLON -38.184802 178.347005 16.554 0
#EQSRCE EV43 LATLON -38.195378 178.315706 16.59 0
#EQSRCE EV44 LATLON -38.189042 178.372119 16.152 0
#EQSRCE EV45 LATLON -38.18842 178.371208 16.349 0
#EQSRCE EV46 LATLON -38.190324 178.315918 16.908 0
#EQSRCE EV47 LATLON -38.481514 178.069124 17.073 0   

EQSTA PRGZ   P      GAU  0.1    GAU  0.1
EQSTA PRGZ   S      GAU  0.5    GAU  0.5
EQSTA RIGZ   P      GAU  0.1    GAU  0.1
EQSTA RIGZ   S      GAU  0.5    GAU  0.5
EQSTA KNZ   P      GAU  0.1    GAU  0.1
EQSTA KNZ   S      GAU  0.5    GAU  0.5
EQSTA MHGZ   P      GAU  0.1    GAU  0.1
EQSTA MHGZ   S      GAU  0.5    GAU  0.5
EQSTA LBPR1   P      GAU  0.1    GAU  0.1
EQSTA SNGZ   P      GAU  0.1    GAU  0.1
EQSTA SNGZ   S      GAU  0.5    GAU  0.5
EQSTA TKGZ   P      GAU  0.1    GAU  0.1
EQSTA TKGZ   S      GAU  0.5    GAU  0.5
EQSTA EBS5   P      GAU  0.1    GAU  0.1
EQSTA EBS5   S      GAU  0.5    GAU  0.5
EQSTA RAGZ   P      GAU  0.1    GAU  0.1
EQSTA RAGZ   S      GAU  0.5    GAU  0.5
EQSTA WHHZ   P      GAU  0.1    GAU  0.1
EQSTA WHHZ   S      GAU  0.5    GAU  0.5
EQSTA CNGZ   P      GAU  0.1    GAU  0.1
EQSTA CNGZ   S      GAU  0.5    GAU  0.5
EQSTA LOBS8   P      GAU  0.1    GAU  0.1
EQSTA LOBS8   S      GAU  0.5    GAU  0.5
EQSTA RAHZ   P      GAU  0.1    GAU  0.1
EQSTA RAHZ   S      GAU  0.5    GAU  0.5
EQSTA MWZ   P      GAU  0.1    GAU  0.1
EQSTA MWZ   S      GAU  0.5    GAU  0.5
EQSTA EBS2   P      GAU  0.1    GAU  0.1
EQSTA EBS2   S      GAU  0.5    GAU  0.5
EQSTA LOBS7   P      GAU  0.1    GAU  0.1
EQSTA LOBS7   S      GAU  0.5    GAU  0.5
EQSTA RTZ   P      GAU  0.1    GAU  0.1
EQSTA RTZ   S      GAU  0.5    GAU  0.5
EQSTA TWGZ   P      GAU  0.1    GAU  0.1
EQSTA TWGZ   S      GAU  0.5    GAU  0.5
EQSTA EBS3   P      GAU  0.1    GAU  0.1
EQSTA EBS3   S      GAU  0.5    GAU  0.5
EQSTA EBS1   P      GAU  0.1    GAU  0.1
EQSTA EBS1   S      GAU  0.5    GAU  0.5
EQSTA MTHZ   P      GAU  0.1    GAU  0.1
EQSTA MTHZ   S      GAU  0.5    GAU  0.5
EQSTA NMHZ   P      GAU  0.1    GAU  0.1
EQSTA NMHZ   S      GAU  0.5    GAU  0.5
EQSTA URZ   P      GAU  0.1    GAU  0.1
EQSTA URZ   S      GAU  0.5    GAU  0.5
EQSTA LOBS6   P      GAU  0.1    GAU  0.1
EQSTA LOBS6   S      GAU  0.5    GAU  0.5
EQSTA EBS4   P      GAU  0.1    GAU  0.1
EQSTA EBS4   S      GAU  0.5    GAU  0.5
EQSTA PUZ   P      GAU  0.1    GAU  0.1
EQSTA PUZ   S      GAU  0.5    GAU  0.5
EQSTA RUGZ   P      GAU  0.1    GAU  0.1
EQSTA RUGZ   S      GAU  0.5    GAU  0.5
EQSTA HAZ   P      GAU  0.1    GAU  0.1
EQSTA HAZ   S      GAU  0.5    GAU  0.5
EQSTA LOBS1   P      GAU  0.1    GAU  0.1
EQSTA LOBS1   S      GAU  0.5    GAU  0.5
EQSTA LOBS2   P      GAU  0.1    GAU  0.1
EQSTA LOBS2   S      GAU  0.5    GAU  0.5
EQSTA LOBS3   P      GAU  0.1    GAU  0.1
EQSTA LOBS3   S      GAU  0.5    GAU  0.5
EQSTA LOBS4   P      GAU  0.1    GAU  0.1
EQSTA LOBS4   S      GAU  0.5    GAU  0.5
EQSTA LOBS5   P      GAU  0.1    GAU  0.1
EQSTA LOBS5   S      GAU  0.5    GAU  0.5
EQSTA LOBS9   P      GAU  0.1    GAU  0.1
EQSTA LOBS9   S      GAU  0.5    GAU  0.5
EQSTA WMGZ   P      GAU  0.1    GAU  0.1
EQSTA WMGZ   S      GAU  0.5    GAU  0.5
EQSTA PKGZ   P      GAU  0.1    GAU  0.1
EQSTA PKGZ   S      GAU  0.5    GAU  0.5


EQQUAL2ERR 0.05 0.1 0.1 0.4 99999.9 99999.9 99999.9 99999.9 99999.9 99999.9

EQVPVS -1
#
#
# =============================================================================
# END of Time2EQ control file statements
# =============================================================================
# =============================================================================





# =============================================================================
# =============================================================================
# NLLoc control file statements
# =============================================================================
#

LOCSIG E. Warren-Smith GNS

LOCCOM 2014-2015 HOBITSS

LOCFILES ./obs/synth_Raukumara_cluster.obs NLLOC_OBS ./time/NZ3D ./loc/NZ3D_Gau_Synth
#LOCFILES ./obs/select_ENRIR.out SEISAN ./time/NZ3D ./loc/NZ3D_Gau_ENRIR

LOCHYPOUT SAVE_NLLOC_ALL SAVE_HYPOINV_SUM SAVE_HYPOELL_SUM

LOCSEARCH  OCT 10 10 10 0.01 10000 5000 1 0
#LOCSEARCH GRID 500
#LOCSEARCH MET 10000 1000 4000 5000 5 -1 0.01 8.0 1.0e-10

LOCGRID 156 361 121 -375 -875 -1.5 5 5 5 PROB_DENSITY SAVE
#LOCGRID 100 100 100 -1.0e30 -1.0e30 0.0 5 5 5 PROB_DENSITY SAVE
#
#GridSearch#LOCGRID  51 51 21  -100.0 -100.0 0.0  4.0 4.0 1.0   PROB_DENSITY  NO_SAVE
#GridSearch#LOCGRID  51 51 21  -1.0e30 -1.0e30 0.0  0.5 0.5 1.0   MISFIT  NO_SAVE
#GridSearch#LOCGRID  81 81 81  -1.0e30 -1.0e30 0.0  0.25 0.25 0.25  PROB_DENSITY  SAVE

LOCMETH GAU_ANALYTIC 9999.0 4 400 -1 -1 -1 -1 1
#LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1 6 -1.0 1
#LOCMETH GAU_ANALYTIC 9999.0 4 -1 -1 -1 6
#LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1 6 -1.0 1


# gaussian model error parameters
# (LOCGAU Sigma_T (s), CorrLen (km))
LOCGAU 0.0 20.0

# travel-time dependent gaussian model error parameters
# (LOCGAU2 SigmaTfraction,  SigmaTmin (s),  SigmaTmax (s))
# travel time error is travel_time*SigmaTfraction, with max/min value = SigmaTmin/SigmaTmax
#LOCGAU2 0.05 0.05 2.0

LOCPHASEID  P   P p G PN PG
LOCPHASEID  S   S s G SN SG


# quality to error mapping (for HYPO71, etc)
# (LOCQUAL2ERR Err0 Err1 Err2 ... )
#
# the following quality mapping is default from Hypoellipse documentation
LOCQUAL2ERR 0.05 0.1 0.2 0.6 99999.9 99999.9 99999.9 99999.9 99999.9 99999.9

# phase statistics parameters
# (LOCPHSTAT RMS_Max, NRdgs_Min, Gap_Max, P_ResMax, S_ResMax)
#    (float)   RMS_Max : max hypocenter RMS to include in ave res
#    (float)   NRdgs_Min : min hypocenter num readings to include in ave res
#    (float)   Gap_Max : max hypocenter gap (deg) to include in ave res
#    (float)   P_ResMax : max abs(P res) to include in ave res
#    (float)   S_ResMax : max abs(S res) to include in ave res
#    (float)   S_ResMax : max abs(S res) to include in ave res
#    (float)   Ell_Len3_Max : max ellipsoid major semi-axis length to include in ave res
#    (float)   Hypo_Depth_Min : min hypo depth to include in ave res
#    (float)   Hypo_Depth_Max : max hypo depth to include in ave res
LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9

LOCANGLES ANGLES_YES 5

LOCMAG ML_HB 1.0 1.0 0.0029


LOCEXCLUDE URZ P
LOCEXCLUDE URZ S
LOCEXCLUDE HAZ P
LOCEXCLUDE HAZ S
LOCEXCLUDE PKGZ P
LOCEXCLUDE PKGZ S
LOCEXCLUDE WHHZ S
LOCEXCLUDE BKZ P
LOCEXCLUDE BKS S
LOCEXCLUDE PUZ P
LOCEXCLUDE PUZ S


#
#
# =============================================================================
# END of NLLoc control file statements
# =============================================================================
# =============================================================================

