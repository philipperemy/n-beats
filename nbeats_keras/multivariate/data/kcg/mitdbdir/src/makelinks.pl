#!/usr/bin/perl
# file: makelinks.pl		G. Moody	30 October 1996
#
# This program rewrites an HTML file of MIMIC database clinical data, adding
# links to wavescript .xws files (also generated here) to any flagged date/time
# stamps found in the input file.  The `@' character should be placed before
# and after each date/time stamp in order to flag it for processing.  The
# rewritten HTML appears on the standard output.
#
# Note: date/time stamps in the input should be in hh:mm:ss dd/mm/yyyy format.

while (<>) {
    if (/\d{3} /) {
	chop();
	@fields = split(/ /, $_);
	$record = shift(@fields);
	$tstamp = shift(@fields);
	if (! $tstamp =~ /\d/) {
	    $tstamp = shift(@fields);
	}
	@tparts = split(/:/, $tstamp);
	$minute = shift(@tparts);
	$second = shift(@tparts);
	$xws = sprintf("samples/%3d%02d%02d.xws", $record, $minute, $second);
	print "<a href=\"$xws\">$tstamp</a>";
	while ($word = shift(@fields)) {
	    print " $word";
	}
	print "<br>\n";
	open(XWS, ">$xws");
	print XWS "-r $record\n";
	print XWS "-a atr\n";
	print XWS "-f $minute:$second\n";
	close(XWS);
    } else {
	print;
    }
}
