@echo off
: file: printdir.bat	G. Moody	11 August 1992

: MS-DOS batch file for printing the MIT-BIH Arrhythmia Database Directory

: Copyright(C) Massachusetts Institute of Technology 1992. All rights reserved.

echo If you have a PostScript printer, you can use this program to print a copy
echo of the MIT-BIH Arrhythmia Database Directory on it.  It will take about 10
echo hours to print the entire 200-page directory.  Make sure that the current
echo drive contains the MIT-BIH Arrhythmia Database CD-ROM, and that the TEMP
echo variable names a writable directory with at least 15 Mb free.  The current
echo value of TEMP is %TEMP%.

echo If you want to do anything other than print the entire directory, see the
echo UNIX 'makefile' in this directory;  it can be used with Microsoft or
echo Borland 'make' with only minor changes.
pause

echo This program needs to change the DB and PATH variables.  When it finishes,
echo you may wish to reset them to their current values, which are
echo DB=%DB%
echo PATH=%PATH%
pause

: DB gets cleared out, so that 'psfd' and 'pschart' use their built-in default
: DB paths to find the records.  This works only if the CD-ROM is in the
: current drive.
set DB=

: The 'bin' directory of the CD-ROM goes at the front of the new PATH, so that
: the correct versions of 'gzip', 'pschart', and 'psfd' will be found.
set PATH=\bin;%PATH%

echo OK, let's see if your printer can print a test page.  This may take
echo up to 5 minutes ... when it's done, you should get a plot with a 12-second
echo ECG strip at the top of a (mostly empty) page.  If you get lots of text
echo instead, you may need to modify this batch file.  CHECK THE OUTPUT BEFORE
echo CONTINUING.

pschart -b 5 -e -g -l -r -R -V -n 999 -T "Test Page" extest >%TEMP%\ex.ps
print %TEMP%\ex.ps
pause

echo Now printing the directory ...

: outside cover
copy cover.z %TEMP%
gzip -d %TEMP%\cover
print %TEMP%\cover

: title page, copyright notice
copy title.z %TEMP%
gzip -d %TEMP%\title
print %TEMP%\title

: table of contents
copy contents.z %TEMP%
gzip -d %TEMP%\contents
print %TEMP%\contents

: foreword, introduction, table of symbols
copy intro.z %TEMP%
gzip -d %TEMP%\intro
print %TEMP%\intro

: introduction to section of full disclosures
copy fdtext.z %TEMP%
gzip -d %TEMP%\fdtext
print %TEMP%\fdtext

: annotated full disclosure plots
: If disk space is low, you might try sending the output directly to PRN
psfd -b 5 -e -g -l -R -V -x -1 -n 2 -T "MIT-BIH Arrhythmia Database" fdlist >%TEMP%\fd.ps
print %TEMP%\fd.ps

: introduction to section of example strips
copy extext.z %TEMP%
gzip -d %TEMP%\extext
print %TEMP%\extext

: annotated example strips
: If disk space is low, you might try sending the output directly to PRN
pschart -b 5 -e -g -l -r -R -V -n 99 -T Examples exlist >%TEMP%\ex.ps
print %TEMP%\ex.ps

: demographic and statistical summaries
copy notes.z %TEMP%
gzip -d %TEMP%\notes
print %TEMP%\notes

: tables of beats and rhythms
copy tables.z %TEMP%
gzip -d %TEMP%\tables
print %TEMP%\tables

: index of full disclosures, examples, notes
copy index.z %TEMP%
gzip -d %TEMP%\index
print %TEMP%\index

echo The entire directory has now been printed.  Temporary files will be
echo removed from %TEMP% next.
pause

del %TEMP%\cover
del %TEMP%\title
del %TEMP%\contents
del %TEMP%\intro
del %TEMP%\fdtext
del %TEMP%\fd.ps
del %TEMP%\extext
del %TEMP%\ex.ps
del %TEMP%\notes
del %TEMP%\tables
del %TEMP%\index
