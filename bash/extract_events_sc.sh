#!/usr/bin/bash

# Pull event ids in time range from default db
scevtls -d localhost --begin "2021-06-01 00:00:00" --end "2022-03-01 00:00:00" > evids.txt
# Another option if more information needed; Use scquery and extract ids
# scquery -d localhost eventByAuthor 38.76 38.93 -118.43 -118.23 1 100 -3 5 "2010-01-01 00:00:00" "2024-05-09 00:00:00" scolv@chopp-precision-5820 | cut -d '|' -f1 > evids.txt
# Loop over each id and write an xml and miniseed for each
while read p; do
  # XML
  scxmldump -fPAMF -E "$p" -o "${p}.xml" -d localhost
  # Get list of streams and times for this evt, then pipe to scart for wav extraction
  scevtstreams -E "$p" -d localhost -m 120 | scart -dsvE --list - /Data2/AmplifyEGS/scarchive > "${p}.ms"
done < evids.txt
