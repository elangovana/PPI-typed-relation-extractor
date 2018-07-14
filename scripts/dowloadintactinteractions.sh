#!/usr/bin/env bash
#!/bin/sh
HOST='ftp.ebi.ac.uk'
USER='anonymous'
PASSWD='anonymous'
FILE=/pub/databases/intact/current/psi25/species/human*.xml
mkdir data
LDIR=data
ftp -n $HOST <<END_SCRIPT
quote USER $USER
quote PASS $PASSWD
prompt noprompt
lcd $LDIR
cd /pub/databases/intact/current/psi25/species
mget human*.xml
quit
END_SCRIPT