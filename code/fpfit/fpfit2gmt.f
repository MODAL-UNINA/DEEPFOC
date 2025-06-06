
c	reformats y2k fpfit summary files into format suitable for GMT psmeca
c	reads/writes to standard input

c	eg., fpfit2gmt <fps.sum >gmt.out

	character*139 line
	character*1 lat_sign, lon_sign
	real latd, latmin, lond, lonmin, lat, lon
	real depth, strike, dip, rake, mag
100	read (5, '(a)', end = 1000) line
	read (line, '(t20, f3.0)') latd
	read (line, '(t23, a1)') lat_sign
	read (line, '(t24, f5.2)') latmin
	read (line, '(t29, f4.0)') lond
	read (line, '(t33, a1)') lon_sign
	read (line, '(t34, f5.2)') lonmin
	lat = latd + latmin/60.
	if (lat_sign .eq. 'S') lat = -lat
	lon = lond + lonmin/60.
	if (lon_sign .ne. 'E') lon = -lon
	read (line, '(t39, f7.2)') depth
	read (line, '(t84, f3.0)') strike
	strike = strike - 90.
	if (strike .lt. 0.) strike = strike + 360.
	read (line, '(t88, f2.0)') dip
	read (line, '(t90, f4.0)') rake
	read (line, '(t48, f5.2)') mag
	write (6, 200) lat, lon, depth, strike, dip, rake, mag, '0. 0.'
200	format (f7.4, 1x, f9.4, 1x, f5.1, 4(f5.0, 1x), a)
	goto 100
1000	stop
	end
