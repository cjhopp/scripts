# Relocate MF detections for Patua with lag-calc picks
scanloc -vv --ep Patua_dets_MAD10_w-mags-1amp-cc07_Mw-preferred_SC3ML.xml -d localhost --log-file scanloc_patua-MF.log --cluster-search-log-file cluster_patua-MF.log > origins_patua-MF.xml
scevent -vv --ep origins_patua-MF.xml -d localhost --log-file scevent_patua-MF.log > events_Patua-MF.xml
