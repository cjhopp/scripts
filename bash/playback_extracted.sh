DIR=/home/sysop/chet-meq/newberry/exracted_events/
echo Process miniseeds in $DIR

for f in *DIR/*.ms; do
  fname=$(basename -- "$f")
  eid="${fname%.*}"
  scautopick -v -I "$f" --playback --ep -d localhost --log-file "${eid}_scautopick.log" > "${eid}_picks.xml"
  ## Working commands for Patua pulling configuration from database
  scanloc -v --ep "${eid}_picks.xml" -d localhost --log-file "${eid}_scanloc.log" --cluster-search-log-file "${eid}_cluster.log" > "${eid}_origins.xml"
  scamp -v --ep "${eid}_origins.xml" -d localhost -I sdsarchive:///Data2/AmplifyEGS/scarchive --log-file "${eid}_scamp.log" > "${eid}_amps.xml"
  scmag -v --ep "${eid}_amps.xml" -d localhost --log-file "${eid}_scmag.log" > "${eid}_mags.xml"
  scevent -v --ep "${eid}_mags.xml" -d localhost --log-file "${eid}_scevent.log" > "${eid}_events.xml"
done
