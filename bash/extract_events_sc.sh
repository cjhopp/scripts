#!/usr/bin/bash

# Pull event ids in time range from default db
scevtls -d localhost --begin "2021-06-01 00:00:00" --end "2022-03-01 00:00:00" > evids.txt
# Loop over each id and write an xml and miniseed for each
while read p; do
  # XML
  scxmldump -fPAMF -E "$p" -o "${p}.xml" -d localhost
  # Get list of streams and times for this evt, then pipe to scart for wav extraction
  scevtstreams -E "$p" -d localhost -m 120 | scart -dsvE --list - /Data2/AmplifyEGS/scarchive > "${p}.ms"
done < evids.txt
