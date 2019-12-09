# file: makefile	G. Moody	22 August 1988
#			Last revised:	  19 May 1997
# 'make' description file for printing MIT-BIH Arrhythmia Database Directory
#
# Copyright(C) Massachusetts Institute of Technology 1997. All rights reserved.

# This file is used with the UNIX `make' command to print a copy of the MIT-BIH
# Arrhythmia Database Directory.  To do so, you will need a PostScript printer,
# the GNU `gzip' utility (see ../../software/gzip124.tar), `pschart' and `psfd'
# (sources are in `../../software/db/app'), about 220 sheets of paper, and
# about 15 Mb for working space in the file system to which TMP (see below)
# belongs.  The directory containing the MIT-BIH Arrhythmia Database must be
# named in the DB environment variable.

# If your printer is an Apple LaserWriter or equivalent, it will take about 10
# hours to print the entire directory (and you will need to reload the paper
# tray after about 5 hours).  In theory, at least, a Linotronic phototypesetter
# should be able to do the job in about 20 hours (send me a copy of the output
# if you try this).  Using a Sun SPARCprinter, the directory can be printed in
# about 45 minutes.  Your printer will not be busy throughout this period;  it
# will take a significant amount of time to generate the PostScript files to be
# printed.

# To recreate the compressed PostScript files (for example, if you wish to
# change the page format), you will also need `troff' and the `ms' macro
# package, `tbl', a troff-to-PostScript translator (e.g., GNU `groff' or Adobe
# `ptroff'), the System V `cut' command, and a TeX DVI-to-PostScript translator
# (such as `dvips', included in the UNIX TeX distribution).

# Many (perhaps all) versions of the troff `ms' macros will generate a blank
# page at the beginning or end of each section.  Be sure to pull out the blank
# sheets from the printer output before trying to make two-sided photocopies.
# (Look for them between pages viii and 1, 97 and 98, 177 and 178, 188 and 189,
# and 193 and 194;  there should be a blank sheet following the table of
# contents, however.)  If your printer is capable of two-sided printing, I
# suggest printing each section separately;  note in particular that the
# section of full disclosure plots begins on a left-hand page (so that each
# record can be viewed in its entirety without turning the page).

# Before using this file for the first time, check that the site-specific
# variables below are appropriate for your system.  If you need to change any
# of them, you may either:
#   1. copy this entire directory to a writable file system, and edit this
#      file, or
#   2. use command-line options to reset variables when running `make' (for
#      example, `make "PRINT=bogoprint -with -lots -of -options" ...' resets
#      the value of the PRINT variable;  see your `make' documentation)
# Print the test page by typing `make test';  if you get text rather than a
# plot, add the `-u' option to variables CO and FO (see below) and try again.
# Once you have successfully printed the test page, print the directory, by
# typing `make directory'.

# Printed copies of the directory may also be obtained from MIT (see file
# `../../ORDER.4M' for information).

# Site-specific variables.

# PRINT is the command that prints Postscript files.  If possible, avoid
# copying the files to a spooling directory -- this will save a little time
# and a lot of space.
PRINT = lpr -s

# TMP is the name of a directory in which temporary files can be written.
# The full disclosure plots and the examples are each about 7 Mb;  make sure
# that TMP's file system has at least 15 Mb free if you are attempting to
# print the entire directory in one operation.
TMP = /tmp

# CO specifies options for `pschart'.  You might want to remove the `-e' option
# if you don't intend to make two-sided copies.  Add a `-d' option if your
# printer's resolution is higher than 300 dpi (e.g., add `-d 600' for a 600 dpi
# printer).  Some UNIX print spoolers cannot deal properly with `structured'
# PostScript -- if the test page contains text rather than plots, add a `-u'
# option (and be prepared to re-order the output if your printer delivers it
# printed side down).  See the `man' page for `pschart' (in
# `../../software/db/doc/pschart.1', or the ECG Database Applications Guide)
# for details.
CO = -b 5 -e -g -l -r -R -V

# FO specifies options for `psfd'.  As above, you might want to remove
# the `-e' option, or add a `-d' or `-u' option.
FO = -b 5 -e -g -l -R -V -x -1

# Uncomment the next line if your printer stacks output printed side up.
SECTIONS = index tables notes ex extext fd fdtext intro contents title cover
# Uncomment the next line if your printer stacks output printed side down.
#SECTIONS = cover title contents intro fdtext fd extext ch notes tables index

# .............................................................................
# Ignore these variables unless you want to regenerate the compressed
# PostScript files in this directory.

# DVITPS is the name of the TeX DVI-to-PostScript translator.
# `make title' only).
DVITPS = dvitps

# TROFF is the name of the troff-to-PostScript translator.
TROFF = ptroff -t

# TMS is the TROFF option needed to load the `ms' macro package.  This should
# not need to be changed unless your system is non-standard.
TMS = -ms

# It should not be necessary to modify anything below this line.
# -----------------------------------------------------------------------------

ZFILES = cover.gz title.gz contents.gz intro.gz fdtext.gz extext.gz \
 notes.gz tables.gz index.gz

.IGNORE:

# `make' or `make explain': describe this file
explain:
	@echo If you have a PostScript printer, you can print a copy of the
	@echo MIT-BIH Arrhythmia Database Directory by typing:
	@echo "    make directory"
	@echo Before attempting to doing so, read "'Makefile'" carefully.

# `make test': print a test page
test:
	pschart $(CO) -n 999 -T "Test Page" extest >$(TMP)/ex.ps
	$(PRINT) $(TMP)/ex.ps

# `make directory': print the entire directory
directory:	$(SECTIONS) clean

# `make psz':  regenerate compressed PostScript files
psz:		$(ZFILES)
	
# `make' targets for printing each section of the directory

cover:				# outside cover
	gzip -d <cover.gz | $(PRINT)
cover.gz:	cover.tr
	$(TROFF) cover.tr | gzip >cover.gz

title:				# title page, copyright notice
	gzip -d <title.gz | $(PRINT)
title.gz:	title.dvi
	dvips title.dvi | gzip >title.gz
	$(DVIPRINT) title.dvi

contents:			# table of contents
	gzip -d <contents.gz | $(PRINT)
contents.gz:	contents.tr
	tbl contents.tr | $(TROFF) | gzip >contents.gz

intro:				# foreword, introduction, table of symbols
	gzip -d <intro.gz | $(PRINT)
intro.gz:
	tbl intro.tr | $(TROFF) $(TMS) | gzip >intro.gz

fdtext:				# introduction to section of full disclosures
	gzip -d <fdtext.gz | $(PRINT)
fdtext.gz:	fdtext.tr
	$(TROFF) $(TMS) fdtext.tr | gzip >fdtext.gz

fd:				# annotated full disclosure plots
	psfd $(FO) -n 2 -T "MIT-BIH Arrhythmia Database" fdlist >$(TMP)/fd.ps
	$(PRINT) $(TMP)/fd.ps

extext:				# introduction to section of example strips
	gzip -d <extext.gz | $(PRINT)
extext.gz:	extext.tr
	$(TROFF) $(TMS) extext.tr | gzip >extext.gz

ex:				# annotated example strips
	pschart $(CO) -n 99 -T Examples exlist >$(TMP)/ex.ps
	$(PRINT) $(TMP)/ex.ps

notes:				# demographic and statistical summaries
	gzip -d <notes.gz | $(PRINT)
notes.gz:	$(TMP)/notes.tr
	tbl $(TMP)/notes.tr | $(TROFF) $(TMS) | gzip >notes.gz
$(TMP)/notes.tr:	$(TMP)/dbnotes	# generate tbl/troff file of notes
	cp notes.h $(TMP)/notes.tr
	chmod u+w $(TMP)/notes.tr
	domit "$(TMP)/dbnotes atruth" >>$(TMP)/notes.tr
$(TMP)/dbnotes:		dbnotes.c	# notes file generating program
	cp dbnotes.c $(TMP)
	cd $(TMP); cc -o dbnotes -O dbnotes.c -ldb
	rm -f $(TMP)/dbnotes.c

tables:				# tables of beats and rhythms
	gzip -d <tables.gz | $(PRINT)
tables.gz:	$(TMP)/tables.tr
	tbl $(TMP)/tables.tr | $(TROFF) $(TMS) | gzip >tables.gz
$(TMP)/tables.tr:	$(TMP)/dbtab	# generate tbl/troff file of tables
	sh ./domit "$(TMP)/dbtab atruth" >$(TMP)/dbt.out
	cp tab.1 $(TMP)/tables.tr
	chmod u+w $(TMP)/tables.tr
	cut -f1-18 <$(TMP)/dbt.out >>$(TMP)/tables.tr
	cat tab.2 >>$(TMP)/tables.tr
	cut -f1,19-33 <$(TMP)/dbt.out | tr -d " " >>$(TMP)/tables.tr
	sh ./domit "$(TMP)/dbtab atruth" -t >$(TMP)/dbt.out
	cat tab.3 >>$(TMP)/tables.tr
	cut -f1-18 <$(TMP)/dbt.out >>$(TMP)/tables.tr
	cat tab.4 >>$(TMP)/tables.tr
	cut -f1,19-33 <$(TMP)/dbt.out | tr -d " " >>$(TMP)/tables.tr
	cat tab.5 >>$(TMP)/tables.tr
	rm -f $(TMP)/dbt.out

$(TMP)/dbtab:		dbtab.c		# table generating program
	cp dbtab.c $(TMP)
	cd $(TMP); cc -o dbtab -O dbtab.c -ldb
	rm -f $(TMP)/dbtab.c

index:				# index of full disclosures, examples, notes
	gzip -d <index.gz | $(PRINT)
index.gz:
	tbl index.tr | $(TROFF) $(TMS) | gzip >index.gz


# `make clean': remove output files from TMP
clean:
	cd $(TMP); rm -f fd.ps ex.ps notes.tr tables.tr dbnotes dbtab
