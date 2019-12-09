/* file: dbtab.c	G. Moody	25 August 1988
			Last revised:	22 July 1992
Add a line to the database summary table

Copyright (C) Massachusetts Institute of Technology 1992. All rights reserved.

This program generates tabular data that are eventually stuffed into a
tbl/troff format file to be printed in the MIT-BIH Arrhythmia Database
Directory.  The code is *not* pretty.  Sorry about that...
*/

#include <stdio.h>
#include <ecg/db.h>
#include <ecg/ecgmap.h>

/* Rhythm codes.  These may be rearranged arbitrarily, but the table rstring[]
   must be kept in agreement.  The order of these codes determines the order
   in which rhythms are printed in the rhythm table. */
#define NSR	1
#define SBR	2
#define BII	3
#define PREX	4
#define AB	5
#define SVTA	6
#define AFL	7
#define AFIB	8
#define PR	9
#define NOD	10
#define VBG	11
#define VTG	12
#define IVR	13
#define VT	14
#define VFL	15

#define MAXR	15	/* largest assigned rhythm code */

static char *rstring[] = {	/* rhythm strings in DB annotation files */
	"",	"N",	"SBR",	"BII",	"PREX",
	"AB",	"SVTA",	"AFL", "AFIB",	"P",
	"NOD",	"B",	"T",	"IVR",	"VT",
	"VFL" };

/* Beat tables and counters: the number of beats of annotation type i during
   the learning period (first 5 minutes) is ltab[i], and during the test period
   (the remainder of the record) is ttab[i];  bstats() accumulates the sum of
   all elements of ltab[] in lsum, and the sum of all elements of ttab[] in
   tsum. */
static long ltab[ACMAX+1], ttab[ACMAX+1], lsum, tsum;
static int tflag = 0;

main(argc, argv)
int argc;
char *argv[];
{
    int i, rhythm = 0;
    long r0 = 0L;
    static struct siginfo si[MAXSIG];
    struct anninfo af[1];
    struct ann annot;
    static char *tape;
    static long rtab[MAXR+2], rtime[MAXR+2], ttest;

/* Check command-line arguments. */
    if (argc < 3) {
	fprintf(stderr, "usage: %s annotator tape [-t]\n", argv[0]);
	exit(1);
    }

/* Open the database files. */
    af[0].name = argv[1]; af[0].stat = READ; tape = argv[2];
    if (dbinit(tape, af, 1, si, MAXSIG) < 0)
	exit(2);

/* Initialize variables. */
    ttest = strtim("5:0");
    rtime[0] = strtim("0.5");
    for (i = 1; i <= MAXR; i++)
	rtime[i] = rtime[0];

/* If specified, skip the learning period. */
    if (argc > 3 && strcmp("-t", argv[3]) == 0) {
	tflag = 1;
	r0 = ttest;
    }
/* Read the annotation file. */
    while (getann(0, &annot) >= 0) {

    /* Count the annotation in the test or learning period as appropriate. */
	if (annot.time >= ttest) ttab[annot.anntyp]++;
	else ltab[annot.anntyp]++;

	if (annot.anntyp == RHYTHM) {
	    if (rhythm) {	/* This condition is true except for the first
				   rhythm label.  For the second and subsequent
				   rhythm labels, update the number of episodes
				   and duration of the rhythm just ended. */
		if (annot.time > r0) {
		    rtab[rhythm]++;
		    rtime[rhythm] += annot.time - r0;
		    r0 = annot.time;
		}
	    }
	    /* Identify the new rhythm. */
	    for (i = 1, rhythm = MAXR+1; i <= MAXR; i++)
		if (strcmp(annot.aux+2, rstring[i]) == 0) {
		    rhythm = i;
		    break;
		}
	}
    }

/* At the end of the annotation file, adjust the counter for the current
   rhythm. */
    if (si[0].nsamp > annot.time) annot.time = si[0].nsamp;
    if (rhythm) {
	rtab[rhythm]++;
	rtime[rhythm] += annot.time - r0;
    }

    printf("%s", tape);

/* Print the beat table. */
    bstats(NORMAL);
    bstats(LBBB);
    bstats(RBBB);
    bstats(APC);
    bstats(ABERR);
    bstats(NPC);
    bstats(SVPB);
    bstats(PVC);
    bstats(FUSION);
    bstats(FLWAV);
    bstats(AESC);
    bstats(NESC);
    bstats(VESC);
    bstats(PACE);
    bstats(PFUS);
    bstats(NAPC);
    bstats(UNKNOWN);

/* Print the rhythm table. */
    for (i = 1; i <= MAXR; i++)
	if (rtab[i] != 0L)
	    printf("\t%s", timstr(rtime[i]));
	else
	    printf("\t\\-");
    printf("\n");
    exit(0);
}


bstats(i)  /* print an entry in the beat table */
int i;
{
    if (tflag) ltab[i] = 0L;
    if (ltab[i] > 0L || ttab[i] > 0L)
	printf("\t%ld", ltab[i]+ttab[i]);
    else printf("\t\\-");
}
