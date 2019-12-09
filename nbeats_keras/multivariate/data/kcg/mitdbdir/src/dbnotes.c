/* file: dbnotes.c	G. Moody	25 August 1988
			Last revised:	6 December 1990
Create database notes file (tbl/troff input format)

Copyright (C) Massachusetts Institute of Technology 1990. All rights reserved.
*/

#include <stdio.h>
#include <ecg/db.h>
#include <ecg/ecgmap.h>

/* Rhythm types.  These may be rearranged arbitrarily;  the order determines
   the order in which rhythms are printed in the rhythm table.  Add a call to
   `setrtab' and adjust N_RTYPES whenever another rhythm is added. */
#define R_UNDEF	0
#define NSR	1
#define SBR	2
#define BII	3
#define B22	4
#define B3	5
#define SAB	6
#define PREX	7
#define AB	8
#define EAR	9
#define SVTA	10
#define AFL	11
#define AFIB	12
#define PR	13
#define NOD	14
#define VBG	15
#define VTG	16
#define IVR	17
#define VT	18
#define VFL	19

#define N_RTYPES	20	/* number of rhythm types, including R_UNDEF */

/* ST and T change types. */
#define STP	0	/* increase in ST level */
#define STN	1	/* decrease in ST level */
#define TP	2	/* increase in T amplitude */
#define TXP	3	/* extreme increase in T amplitude */
#define TN	4	/* decrease in T amplitude */
#define TXN	5	/* extreme decrease in T amplitude */
#define TBP	6	/* biphasic T-wave */

#define N_STYPES	7	/* number of ST and T change types */


#define PLUS	1	/* positive change */
#define MINUS	2	/* negative change */
#define XPLUS	3	/* extreme positive change */
#define XMINUS	4	/* extreme negative change */
#define BIPHASIC 5	/* change to biphasic form */

#define STCH0	1	/* beginning of ST change episode */
#define STCH1	2	/* extremum of ST change */
#define STCH2	3	/* end of ST change episode */

#define TCH0	4	/* beginning of T change episode */
#define TCH1	5	/* extremum of T change */
#define TCH2	6	/* end of T change episode */


/* Noise type bit masks. */
#define C0		0x00	/* signal 0 clean */
#define C1		0x00	/* signal 1 clean */
#define C2		0x00	/* signal 2 clean */
#define C3		0x00	/* signal 3 clean */
#define N0	     	0x01	/* signal 0 noisy */
#define N1	     	0x02	/* signal 1 noisy */
#define N2	     	0x04	/* signal 2 noisy */
#define N3	     	0x08	/* signal 3 noisy */
#define U0	     	0x11	/* signal 0 unreadable */
#define U1	     	0x22	/* signal 1 unreadable */
#define U2		0x44	/* signal 2 unreadable */
#define U3	   	0x88	/* signal 3 unreadable */
#define UNREADABLE   	0xff	/* all signals unreadable */

#define N_NTYPES	256	/* number of noise types */
#define N_RLEN	20		/* number of distinct run lengths */

/* Values for info_type. */
#define NO_INFO	0
#define MITDB	1
#define EDB	2
#define VALEDB	3

struct btabstruct {
    long lcount;	/* number of beats of this type in learning period */
    long tcount;	/* number of beats of this type in test period */
} btab[ACMAX+1];

struct rltabstruct {
    long len;		/* number of beats in run */
    long nruns;		/* number of runs of `len' beats */
} artab[N_RLEN], vrtab[N_RLEN];

struct rtabstruct {
    long episodes;	/* number of episodes of this rhythm */
    long duration;	/* duration of this rhythm (in sample intervals) */
    char *rstring;	/* rhythm string as it appears in annotation file */
    char *rslong;	/* rhythm name as it is printed by this program */
    int min3r, max3r;	/* minimum/maximum length of 3 R-R intervals */
} rtab[N_RTYPES];

struct sttabstruct {
    long episodes;	/* number of episodes of ST or T change of this type */
    long duration;	/* duration of episodes (in sample intervals) */
    char *ststring;	/* description of change as printed by this program */
    int extremum;	/* extreme value of change */
} sttab[N_STYPES][MAXSIG];

struct ntabstruct {
    long episodes;	/* number of episodes of this noise type */
    long duration;	/* duration of this noise type (in sample intervals) */
    char *nstring;	/* noise type as it is printed by this program */
} ntab[N_NTYPES];

long lsum, tsum;	/* numbers of beats of all types in the learning and
			   test periods (tallied by `bstats') */
long ttest;		/* time of the end of the test period */

main(argc, argv)
int argc;
char *argv[];
{
    double t3min;
    int a=0, age, c, i, info_type, magnitude, maxrate, minrate, noise = 0,
	nsig, rhythm = R_UNDEF, rlen = -1, rr3, sign, st_calibrated = 0,
	st_data = 0, t_data = 0, type, v=0;
    static long n0, r0, tq0, tq1, tq2, tq3, s0[N_STYPES][MAXSIG];
    struct siginfo si[MAXSIG];
    struct anninfo ai;
    struct ann annot;
    static char buf[256], *infop, *p, *record, sex[10], umask;

    /* Check command-line arguments. */
    if (argc < 3) {
	fprintf(stderr, "usage: %s annotator record\n", argv[0]);
	exit(1);
    }

    /* Open the database files. */
    ai.name = argv[1]; ai.stat = READ; record = argv[2];
    if (annopen(record, &ai, 1) < 0 ||
	(nsig = isigopen(record, si, -MAXSIG)) < 0)
	exit(2);

    /* Initialize variables. */
    t3min = strtim("3:0");
    ttest = strtim("5:0");

    setrtab(R_UNDEF, NULL, "Unspecified rhythm");
    setrtab(NSR, "N", "Normal sinus rhythm");
    setrtab(SBR, "SBR", "Sinus bradycardia");
    setrtab(BII, "BII", "2\\(de heart block");
    setrtab(B22, "B22", "2\\(de heart block (Mobitz type II)");
    setrtab(B3, "B3", "3\\(de heart block");
    setrtab(SAB, "SAB", "Sino-atrial block");
    setrtab(PREX, "PREX", "Pre-excitation (WPW)");
    setrtab(AB, "AB", "Atrial bigeminy");
    setrtab(EAR, "EAR", "Ectopic atrial rhythm");
    setrtab(SVTA, "SVTA", "SVTA");
    setrtab(AFL, "AFL", "Atrial flutter");
    setrtab(AFIB, "AFIB", "Atrial fibrillation");
    setrtab(PR, "P", "Paced rhythm");
    setrtab(NOD, "NOD", "Nodal (junctional) rhythm");
    setrtab(VBG, "B", "Ventricular bigeminy");
    setrtab(VTG, "T", "Ventricular trigeminy");
    setrtab(IVR, "IVR", "Idioventricular rhythm");
    setrtab(VT, "VT", "Ventricular tachycardia");
    setrtab(VFL, "VFL", "Ventricular flutter");

    for (c = 0; c < nsig; c++) {
	setstab(STP, c, "Positive ST dev.");
	setstab(STN, c, "Negative ST dev.");
	setstab(TP, c, "Positive T dev.");
	setstab(TXP, c, "T dev. \\(>= 400 \\(*mV");
	setstab(TN, c, "Negative T dev.");
	setstab(TXN, c, "T dev. \\(<= \\(mi400 \\(*mV");
	setstab(TBP, c, "Biphasic T");
    }

    switch (nsig) {
      case 1:
	setntab(C0, "Clean");
	setntab(N0, "Noisy");
	setntab(U0, "Unreadable");
	setntab(UNREADABLE, "Unreadable");
	umask = U0;
	break;		/* so far, so good ... */
      case 2:
	setntab(C0|C1, "Both clean");
	setntab(C0|N1, "Lower noisy");
	setntab(C0|U1, "Lower unreadable");
	setntab(N0|C1, "Upper noisy");
	setntab(N0|N1, "Both noisy");
	setntab(N0|U1, "Upper noisy, lower unreadable");
	setntab(U0|C1, "Upper unreadable");
	setntab(U0|N1, "Upper unreadable, lower noisy");
	setntab(U0|U1, "Both unreadable");
	setntab(UNREADABLE, "Unreadable");
	umask = U0|U1;
	break;		/* yikes! */
      case 3:
	setntab(C0|C1|C2, "All clean");
	setntab(C0|C1|N2, "Lower noisy");
	setntab(C0|C1|U2, "Lower unreadable");
	setntab(C0|N1|C2, "Middle noisy");
	setntab(C0|N1|N2, "Middle and lower noisy");
	setntab(C0|N1|U2, "Middle noisy, lower unreadable");
	setntab(C0|U1|C2, "Middle unreadable");
	setntab(C0|U1|N2, "Middle unreadable, lower noisy");
	setntab(C0|U1|U2, "Middle and lower unreadable");
	setntab(N0|C1|C2, "Upper noisy");
	setntab(N0|C1|N2, "Upper and lower noisy");
	setntab(N0|C1|U2, "Upper noisy, lower unreadable");
	setntab(N0|N1|C2, "Upper and middle noisy");
	setntab(N0|N1|N2, "All noisy");
	setntab(N0|N1|U2, "Lower unreadable, others noisy");
	setntab(N0|U1|C2, "Upper noisy, middle unreadable");
	setntab(N0|U1|N2, "Middle unreadable, others noisy");
	setntab(N0|U1|U2, "Upper noisy, others unreadable");
	setntab(U0|C1|C2, "Upper unreadable");
	setntab(U0|C1|N2, "Upper unreadable, lower noisy");
	setntab(U0|C1|U2, "Upper and lower unreadable");
	setntab(U0|N1|C2, "Upper unreadable, middle noisy");
	setntab(U0|N1|N2, "Upper unreadable, others noisy");
	setntab(U0|N1|U2, "Middle noisy, others unreadable");
	setntab(U0|U1|C2, "Upper and middle unreadable");
	setntab(U0|U1|N2, "Lower noisy, others unreadable");
	setntab(U0|U1|U2, "All unreadable");
	setntab(UNREADABLE, "Unreadable");
	umask = U0|U1|U2;
	break;		/* arggggh! */
/*    case 4:		   nonononononononononono!!!!!!! */
      default:
	umask = UNREADABLE;
	break;
    }

    /* Read the annotation file. */
    while (getann(0, &annot) >= 0) {

    /* Count the annotation in the test or learning period as appropriate. */
	if (annot.time >= ttest) btab[annot.anntyp].tcount++;
	else btab[annot.anntyp].lcount++;

	if (isqrs(annot.anntyp)) {

	    /* Update the ectopic activity tables if appropriate. */
	    switch (map2(annot.anntyp)) {
		case NORMAL:	if (a > 0) { aupdate(a); a = 0; }
				if (v > 0) { vupdate(v); v = 0; }
				break;
		case SVPB:	if (v > 0) { vupdate(v); v = 0; }
				a++;
				break;
		case PVC:
		case FUSION:	if (a > 0) { aupdate(a); a = 0; }
				v++;
				break;
	    }

	    /* Keep track of the times of the last 4 beats. */
	    tq0 = tq1; tq1 = tq2; tq2 = tq3; tq3 = annot.time;

	    /* Update the count of consecutive beats in the current rhythm. */
	    rlen++;

	    /* Check various conditions which might make a heart rate
	       measurement invalid. */
	    switch (rhythm) {
	      case NSR:
	      case SBR:
	      case BII:
	      case B22:
	      case B3:
	      case SAB:
	      case PREX: 
		if (map2(annot.anntyp) != NORMAL) rlen = -1; break;
	      case AB:
	      case NOD:
		if (map1(annot.anntyp) != NORMAL) rlen = -1; break;
	      case EAR:
	      case SVTA:
		if (map2(annot.anntyp) != SVPB) rlen = -1; break;
	      case PR:
		if (annot.anntyp != PACE) rlen = -1; break;
	      case IVR:
	      case VT:
		if (map1(annot.anntyp) == NORMAL) rlen = -1; break;
	      case VFL:
		if (annot.anntyp != FLWAV) rlen = -1; break;
	      default:
		break;
	    }	

	    /* If a valid heart rate can be measured, update the records of
	       maximum and minimum rate for the current rhythm.  Note that
	       these can be determined from the minimum and maximum R-R
	       intervals;  what is actually used here is the sum of three
	       consecutive intervals. */
	    if (rlen >= 3) {
		rr3 = tq3 - tq0;
 		if (rr3 > rtab[rhythm].max3r) rtab[rhythm].max3r = rr3;
		if (rr3 < rtab[rhythm].min3r) rtab[rhythm].min3r = rr3;
	    }
	}
	else if (annot.anntyp == RHYTHM) {
	    /* Don't count the interval from the beginning of the record to
	       the first rhythm label as `unspecified rhythm' unless the first
	       beat label occurs before the first rhythm label. */
	    if (tq3 > 0L) {
		rtab[rhythm].episodes++;
		rtab[rhythm].duration += annot.time - r0;
		r0 = annot.time;
	    }
	    /* Identify the new rhythm. */
	    for (i = 1, rhythm = R_UNDEF; i < N_RTYPES; i++)
		if (strcmp(annot.aux+2, rtab[i].rstring) == 0) {
		    rhythm = i;
		    break;
		}
	    if (rhythm == SVTA || rhythm == VT) rlen = 0;
	    else rlen = -1;
	}
	else if (annot.anntyp == STCH || annot.anntyp == TCH) {
	    /* Identify the nature of the ST or T change from the aux field.
	       The syntax of the aux field is one of:
	         "(" <type> [<signal designation>] <sign>
		 "A" <type> [<signal designation>] <sign> [<magnitude>]
		 <type> [<signal designation>] <sign> ")"
	       where <type> is "ST" or "T", <signal designation> is 1 or 2
	       decimal digits, <sign> is "+", "++", "-", "--", or "BP", and
	       <magnitude> is 2-4 decimal digits.  ("++" and "--" appear only
	       in T-wave change annotations;  they designate extreme (>= 0.4 mV
	       changes in T-wave amplitude.)  The signal designation is
	       the signal number (e.g., signal 0 is designated by "0",
	       etc.);  in records which contain only one signal (such as those
	       in the VALE DB), the signal designation is omitted.  The
	       magnitude is given in microvolts, and may also be omitted (as
	       in the VALE DB).
	     */
	    if (annot.aux == NULL)
		fprintf(stderr,
			"warning: %s annotation without text field at %s\n",
			ecgstr(annot.anntyp), timstr(annot.time));
	    else {
		p = annot.aux+1;
		switch (*p) {
		  case '(':
		    if (annot.anntyp == STCH) {
			type = STCH0;
			p += 3;		/* skip `(ST' */
		    }
		    else {
			type = TCH0;
			p += 2;		/* skip `(T' */
		    }
		    break;
		  case 'A':
		    if (annot.anntyp == STCH) {
			type = STCH1;
			p += 3;		/* skip `(ST' */
		    }
		    else {
			type = TCH1;
			p += 2;		/* skip `(T' */
		    }
		    break;
		  case 'S':
		    type = STCH2;
		    p += 2;		/* skip `ST' */
		    break;
		  case 'T':
		    type = TCH2;
		    p++;		/* skip `T' */
		    break;
		  default:
		    fprintf(stderr,
			"warning: %s annotation with text field %s at %s\n",
			ecgstr(annot.anntyp), annot.aux, timstr(annot.time));
		    type = 0;
		    break;
		}
		if (nsig > 1) {
		    if ((c = atoi(p)) > 9) {
			p++;
		    }
		    p++;
		}
		else c = 0;
		switch (*p++) {
		  case '+':
		    if (*p == '+') {
			sign = XPLUS;
			p++;
		    }
		    else
			sign = PLUS;
		    break;
		  case '-':
		    if (*p == '-') {
			sign = XMINUS;
			p++;
		    }
		    else
			sign = MINUS;
		    break;
		  case 'B':
		    if (*p == 'P') {
			sign = BIPHASIC;
			p++;
		    }
		    break;
		  default:
		    fprintf(stderr,
			"warning: %s annotation with text field %s at %s\n",
			ecgstr(annot.anntyp), annot.aux, timstr(annot.time));
		    sign = 0;
		    break;
		}
		if (*p && *p != ')') {
		    if ((magnitude = atoi(p)) > 0)
			st_calibrated = 1;
		}
		else magnitude = 0;
		if (0 <= c && c < MAXSIG) switch (type) {
		  case STCH0:	/* beginning of ST change episode */
		    if (sign == PLUS) {
			st_data = 1;
			s0[STP][c] = annot.time;
		    }
		    else if (sign == MINUS) {
			st_data = 1;
			s0[STN][c] = annot.time;
		    }
		    break;
		  case STCH1:	/* extremum of ST change */
		    if (sign == PLUS && magnitude > sttab[STP][c].extremum)
			sttab[STP][c].extremum = magnitude;
		    else if (sign == MINUS &&
			     magnitude > sttab[STN][c].extremum)
			sttab[STN][c].extremum = magnitude;
		    break;
		  case STCH2:	/* end of ST change episode */
		    if (sign == PLUS) {
			sttab[STP][c].episodes++;
			sttab[STP][c].duration += annot.time - s0[STP][c];
			s0[STP][c] = 0L;
		    }
		    else if (sign == MINUS) {
			sttab[STN][c].episodes++;
			sttab[STN][c].duration += annot.time - s0[STN][c];
			s0[STN][c] = 0L;
		    }
		    break;
		  case TCH0:	/* beginning of T change episode */
		    if (sign == PLUS) {
			t_data = 1;
			s0[TP][c] = annot.time;
		    }
		    else if (sign == MINUS) {
			t_data = 1;
			s0[TN][c] = annot.time;
		    }
		    else if (sign == XPLUS) {
			t_data = 1;
			s0[TXP][c] = annot.time;
		    }
		    else if (sign == XMINUS) {
			t_data = 1;
			s0[TXN][c] = annot.time;
		    }
		    else if (sign == BIPHASIC) {
			t_data = 1;
			s0[TBP][c] = annot.time;
		    }
		    break;
		  case TCH1:	/* extremum of T change */
		    if ((sign == PLUS || sign == XPLUS) &&
			magnitude > sttab[TP][c].extremum)
			sttab[TP][c].extremum = magnitude;
		    else if ((sign == MINUS || sign == XMINUS) &&
			     magnitude > sttab[TN][c].extremum)
			sttab[TN][c].extremum = magnitude;
		    break;
		  case TCH2:	/* end of T change episode */
		    if (sign == PLUS) {
			sttab[TP][c].episodes++;
			sttab[TP][c].duration += annot.time - s0[TP][c];
			s0[TP][c] = 0L;
		    }
		    else if (sign == MINUS) {
			sttab[TN][c].episodes++;
			sttab[TN][c].duration += annot.time - s0[TN][c];
			s0[TN][c] = 0L;
		    }
		    else if (sign == XPLUS) {
			sttab[TXP][c].episodes++;
			sttab[TXP][c].duration += annot.time - s0[TXP][c];
			s0[TXP][c] = 0L;
		    }
		    else if (sign == XMINUS) {
			sttab[TXN][c].episodes++;
			sttab[TXN][c].duration += annot.time - s0[TXN][c];
			s0[TXN][c] = 0L;
		    }
		    else if (sign == BIPHASIC) {
			sttab[TBP][c].episodes++;
			sttab[TBP][c].duration += annot.time - s0[TBP][c];
			s0[TBP][c] = 0L;
		    }
		    break;
		  default:
		    break;
		}
	    }
	}
	else if (annot.anntyp == NOISE) {
	    /* Don't count the interval from the beginning of the record to
	       the first signal quality label as `clean' unless the first beat
	       label occurs before the first signal quality label. */
	    if (tq3 > 0L) {
		ntab[noise].episodes++;
		ntab[noise].duration += annot.time - n0;
		n0 = annot.time;
	    }
	    noise = annot.subtyp & 0xff;

	    /* Unreadable segments invalidate rate measurements. */
	    if ((noise & umask) == umask) rlen = -1;
	}
    }

    /* At the end of the annotation file, count any run in progress and adjust
       the counters for the current rhythm and noise type, and ST and T states.
    */
    if (a) aupdate(a);
    if (v) aupdate(v);
    if (strtim("e") > annot.time) annot.time = strtim("e");
    rtab[rhythm].episodes++;
    rtab[rhythm].duration += annot.time - r0;
    for (c = 0; c < nsig; c++) {
	for (i = 0; i < N_STYPES; i++)
	    if (s0[i][c] > 0L) {
		sttab[i][c].episodes++;
		sttab[i][c].duration += annot.time - s0[i][c];
	    }
	if (sttab[TXP][c].episodes > 0)
	    sttab[TXP][c].extremum = sttab[TP][c].extremum;
	if (sttab[TXN][c].episodes > 0)
	    sttab[TXN][c].extremum = sttab[TN][c].extremum;
    }
    ntab[noise].episodes++;
    ntab[noise].duration += annot.time - n0;

    /* Read the first line of info, if any. */
    infop = getinfo(record);
    if (infop == NULL)
	info_type = NO_INFO;
    else {
	while (*infop == ' ' || *infop == '\t')
	    infop++;
	if (*infop == 'A') {	/* `Age: ...' as in European ST-T DB */
	    sscanf(infop, "Age: %d Sex: %s", &age, sex);
	    info_type = EDB;
	}
	else if (*infop == 'R') {	/* `Reference: ...' as in VALE DB */
	    age = -1; sex[0] = '?';
	    info_type = VALEDB;
	}
	else {			/* fixed format as in MIT DB */
	    sscanf(infop, "%d%s", &age, sex);
	    info_type = MITDB;
	}
    }

    /* Start a "keep" (to force troff to print the following section in a
       single column), and print the section header line. */
    printf(".KS\n.SH\n");

    printf("Record %s  (", record);
    for (i = 0; i < nsig; i++) {
	if (i > 0) fputs(", ", stdout);
	fputs(si[i].desc, stdout);
    }
    fputs("; ", stdout);
    switch (sex[0]) {
      case 'f':
      case 'F':	fputs("female, age ", stdout); break;
      case 'm':
      case 'M': fputs("male, age ", stdout); break;
      default:	fputs("age ", stdout); break;
    }
    if (age > 0) printf("%d)\n", age);
    else fputs("not recorded)\n", stdout);

    switch (info_type) {
      case MITDB:
	/* Print the medications section (the second line of info). */
	if (infop = getinfo(NULL)) {
	    while (*infop == ' ' || *infop == '\t')
		infop++;
	    printf(".LP\n\\fIMedications:\\fR %s\n", infop);
	}
	break;
      default:
	break;
    }

    /* Print the beat table. */
    printf(".TS\nexpand;\nlw(1i) c c c\nl r r r.\n");
    printf("\\fIBeats\tBefore 5:00\tAfter 5:00\tTotal\\fR\n");
    bstats(NORMAL, "Normal");
    bstats(LBBB, "Left BBB");
    bstats(RBBB, "Right BBB");
    bstats(APC, "APC");
    bstats(ABERR, "Aberrated APC");
    bstats(NPC, "Junctional premature");
    bstats(SVPB, "SVPC");
    bstats(PVC, "PVC");
    bstats(RONT, "R-on-T PVC");
    bstats(FUSION, "Fusion PVC");
    bstats(FLWAV, "Ventricular flutter wave");
    bstats(AESC, "Atrial escape");
    bstats(NESC, "Junctional escape");
    bstats(SVESC, "Supraventricular escape");
    bstats(VESC, "Ventricular escape");
    bstats(PACE, "Paced");
    bstats(PFUS, "Pacemaker fusion");
    bstats(NAPC, "Blocked APC");
    bstats(UNKNOWN, "Unclassifiable");
    printf("\\fITotal\t%ld\t%ld\t%ld\\fR\n", lsum, tsum, lsum+tsum);
    printf(".TE\n.KE\n");

    /* Print the supraventricular ectopy table, if necessary. */
    if (artab[0].len > 1 || artab[0].nruns > 1 || artab[1].len > 0) {
	int i, j, k, l;

	printf(".KS\n.TS\nlw(1.125i) l.\n");
	printf("\\fISupraventricular ectopy:\\fR");
        for (i = 0; i < N_RLEN; i++) {
	    for (j = 0, l = 9999; j < N_RLEN; j++)
		if (0 < artab[j].len && artab[j].len < l) l = artab[k = j].len;
	    if (l == 9999) break;
	    switch (l) {
	      case 1:
		printf("\t%d isolated beat%s\n",
		       artab[k].nruns, (artab[k].nruns > 1) ? "s" : "");
		break;
	      case 2:
		printf("\t%d couplet%s\n",
		       artab[k].nruns, (artab[k].nruns > 1) ? "s" : "");
		break;
	      default:
		printf("\t%d run%s of %d beats\n",
		       artab[k].nruns, (artab[k].nruns > 1) ? "s" : "",
		       artab[k].len);
		break;
	    }
	    artab[k].len = 0;
	}
	printf(".TE\n.KE\n");
    }

    /* Print the ventricular ectopy table, if necessary. */
    if (vrtab[0].len > 1 || vrtab[0].nruns > 1 || vrtab[1].len > 0) {
	int i, j, k, l;

	printf(".KS\n.TS\nlw(1.125i) l.\n");
	printf("\\fIVentricular ectopy:\\fR");
        for (i = 0; i < N_RLEN; i++) {
	    for (j = 0, l = 9999; j < N_RLEN; j++)
		if (0 < vrtab[j].len && vrtab[j].len < l) l = vrtab[k = j].len;
	    if (l == 9999) break;
	    switch (l) {
	      case 1:
		printf("\t%d isolated beat%s\n",
		       vrtab[k].nruns, (vrtab[k].nruns > 1) ? "s" : "");
		break;
	      case 2:
		printf("\t%d couplet%s\n",
		       vrtab[k].nruns, (vrtab[k].nruns > 1) ? "s" : "");
		break;
	      default:
		printf("\t%d run%s of %d beats\n",
		       vrtab[k].nruns, (vrtab[k].nruns > 1) ? "s" : "",
		       vrtab[k].len);
		break;
	    }
	    vrtab[k].len = 0;
	}
	printf(".TE\n.KE\n");
    }

    /* Print the rhythm table. */
    printf(".KS\n.TS\nexpand;\nlw c rw(0.4i) rw(0.4i)\nl c n r.\n");
    printf("\\fIRhythm\tRate\tEpisodes\tDuration\\fR\n");
    for (i = 0; i < N_RTYPES; i++)
	if (rtab[i].episodes != 0L) {
	    printf("%s\t", rtab[i].rslong);
	    if (rtab[i].max3r > 0)
		printf("%d", minrate = (int)(t3min/rtab[i].max3r + 0.5));
	    else
		minrate = 0;
	    if (rtab[i].min3r != 0xffff) {
		maxrate = (int)(t3min/rtab[i].min3r + 0.5);
		if (maxrate > minrate) printf("\\-%d", maxrate);
	    }
	    else if (minrate == 0)
		printf("\\-");
	    printf("\t%ld\t%s\n", rtab[i].episodes, timstr(rtab[i].duration));
	}
    printf(".TE\n.KE\n");

    /* Print the ST and T-change table, if available. */
    if (st_data || t_data) {
	printf(".KS\n.TS\nexpand;\nl");
	if (st_calibrated) printf(" r");
	printf(" rw(0.4i) rw(0.4i)\nl");
	if (st_calibrated) printf(" n");
	printf(" n r.\n\\fIST\\-T state");
	if (st_calibrated) printf("\tExtremum");
	printf("\tEpisodes\tDuration\\fR\n");
	for (c = 0; c < nsig; c++) {
	    int header_printed = 0;

	    for (i = 0; i < N_STYPES; i++) {
		if (sttab[i][c].episodes != 0L) {
		    if (header_printed == 0) {
			switch (nsig) {
			  case 1:
			    break;
			  case 2:
			    if (c == 0) printf("\\fI(upper signal)\\fR\n");
			    else printf("\\fI(lower signal)\\fR\n");
			    break;
			  case 3:
			    if (c == 0) printf("\\fI(upper signal)\\fR\n");
			    else if (c == 1)
				printf("\\fI(middle signal)\\fR\n");
			    else printf("\\fI(lower signal)\\fR\n");
			    break;
			  default:
			    printf("(signal %d)\n", c);
			    break;
			}
			header_printed = 1;
		    }
		    printf("%s\t", sttab[i][c].ststring);
		    if (st_calibrated) {
			if (sttab[i][c].extremum > 0) {
			    switch (i) {
			      case STP:
			      case TP:
			      case TXP:	printf("\\(pl"); break;
			      case STN:
			      case TN:
			      case TXN: printf("\\(mi"); break;
			    }
			    printf("%d\t", sttab[i][c].extremum);
			}
			else
			    printf("\\-\t");
		    }
		    printf("%ld\t%s\n", sttab[i][c].episodes,
			   timstr(sttab[i][c].duration));
		}
	    }
	}
	printf(".TE\n.KE\n");
    }

    /* Print the signal quality table. */
    printf(".KS\n.TS\nexpand;\nl rw(0.4i) rw(0.4i)\nl n r.\n");
    printf("\\fISignal quality\tEpisodes\tDuration\\fR\n");
    for (i = 0; i < N_NTYPES; i++)
	if (ntab[i].episodes != 0L) {
	    if (ntab[i].nstring == NULL) {
		static char ns[40];

		sprintf(ns, "Signal quality code %d", i);
		ntab[i].nstring = ns;
	    }
	    printf("%s\t%ld\t%s\n", ntab[i].nstring, ntab[i].episodes,
		   timstr(ntab[i].duration));
	}
    printf(".TE\n");	/* (the "keep" ends after the "notes" section) */

    /* Print a "notes" section if there is any additional info available. */
    switch (info_type) {
      case EDB:
	if (infop = getinfo(NULL)) {
	    while (*infop == ' ' || *infop == '\t')
		infop++;
	    if ('a' <= *infop && *infop <= 'z') *infop += 'A' - 'a';
	    printf(".IP \\fINotes:\\fR .375i\n%s\n", infop);
	    while (infop = getinfo(NULL)) {
		while (*infop == ' ' || *infop == '\t')
		    infop++;
		printf(".br\n%s\n", infop);
	    }
	}
	else
	    printf(".LP\n");
	break;
      case MITDB:
	if (infop = getinfo(NULL)) {
	    while (*infop == ' ' || *infop == '\t')
		infop++;
	    printf(".IP \\fINotes:\\fR .375i\n%s\n", infop);
	    while (infop = getinfo(NULL)) {
		while (*infop == ' ' || *infop == '\t')
		    infop++;
		printf("%s\n", infop);
	    }
	}
	else
	    printf(".LP\n");
	break;
      case VALEDB:
	if (infop = getinfo(NULL)) {
	    while (*infop == ' ' || *infop == '\t')
		infop++;
	    if (strncmp(infop, "Reference:", 10) == 0)
		infop = getinfo(NULL);
	}
	if (infop) {
	    while (*infop == ' ' || *infop == '\t')
		infop++;
	    printf(".IP \\fINotes:\\fR .375i\n%s\n", infop);
	    while (infop = getinfo(NULL)) {
		while (*infop == ' ' || *infop == '\t')
		    infop++;
		printf(".br\n%s\n", infop);
	    }
	}
	else
	    printf(".LP\n");
	break;
      default:
	printf(".LP\n");
	break;
    }
    printf(".KE\n");

/* Close the database files and quit. */
    dbquit();
    exit(0);
}

setrtab(i, rst, rsl)
int i;
char *rst, *rsl;
{
    if (0 <= i && i < N_RTYPES) {
	rtab[i].episodes = 0L;
	rtab[i].duration = strtim("0.5");
	rtab[i].rstring = rst;
	rtab[i].rslong = rsl;
	rtab[i].min3r = 0xffff;
	rtab[i].max3r = 0;
    }
}

setstab(i, signal, sst)
int i, signal;
char *sst;
{
    if (0 <= i && i < N_STYPES && 0 <= signal && signal < MAXSIG) {
	sttab[i][signal].episodes = 0L;
	sttab[i][signal].duration = strtim("0.5");
	sttab[i][signal].ststring = sst;
	sttab[i][signal].extremum = 0;
    }
}

setntab(i, nst)
int i;
char *nst;
{
    if (0 <= i && i < N_NTYPES) {
	ntab[i].episodes = 0L;
	ntab[i].duration = strtim("0.5");
	ntab[i].nstring = nst;
    }
}
	
bstats(i, s)  /* print a line of the beat table if beats of type i were seen */
int i;
char *s;		/* English description of beat type i */
{
    if (btab[i].tcount > 0L || btab[i].lcount > 0L) {
	printf("%s\t", s);
	if (btab[i].lcount > 0L) {
	    printf("%ld\t", btab[i].lcount);
	    lsum += btab[i].lcount;
	}
	else
	    printf("\\-\t");
	if (btab[i].tcount > 0L) {
	    printf("%ld\t", btab[i].tcount);
	    tsum += btab[i].tcount;
	}
	else
	    printf("\\-\t");
	printf("%ld\n", btab[i].lcount + btab[i].tcount);
    }
}

aupdate(i)	/* Record that an i-beat SV run was observed */
int i;
{
    int j;

    for (j = 0; j < N_RLEN-1; j++) {
	if (artab[j].len == i) break;
	else if (artab[j].len == 0) { artab[j].len = i; break; }
    }
    artab[j].nruns++;
}

vupdate(i)	/* Record that an i-beat V run was observed */
int i;
{
    int j;

    for (j = 0; j < N_RLEN-1; j++) {
	if (vrtab[j].len == i) break;
	else if (vrtab[j].len == 0) { vrtab[j].len = i; break; }
    }
    vrtab[j].nruns++;
}
