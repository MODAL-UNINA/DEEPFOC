	subroutine pltnet (xpos1, ypos1, ansc, hite3, event, hite2, pi,
     1 ncfrm, ncdwn, cx1, cx2, cy1, cy2, rmax1, rmax2, ncup, iunit, 
     2 title, ypos2, www)

	implicit none
c
      character*1       ansc                            
c                                                       ! flag: y(n)=do (not) plot color beach balls
      real              cx1                             
c                                                       ! x position of large circle center
      real              cx2                             
c                                                       ! x position of small circle center
      real              cy1                             
c                                                       ! y position of large circle center
      real              cy2                             
c                                                       ! y position of small circle center
      real              depth
c							! depth
      character*132     event                           
c                                                       ! hypo71 summary card
      real		fit90
c							! misfit  + 90% confidence estimate
      real              hite2                           
c                                                       ! height of p,t symbol in small circle, dscrpncy rprt
      real              hite3                           
c                                                       ! height of title, extended summary card, first motion legend
      integer           hunit
c                                                       ! logical unit of www parameter file
      integer           ida
c							! day
      integer           ihr
c							! hour
      integer           imin
c							! minute
      integer           imo
c							! month
      integer           ios
c							! iostat variable
      integer           iunit
c                                                       ! logical unit ".pol" file
      integer           iudip
c							! dip uncertainty
      integer           iurake
c							! rake uncertainty
      integer           iustrk
c							! strike uncertainty
      integer           iyr
c							! year
      character*1       latc
c							! north/south character
      integer           latd                           
c							! latitude degrees
      character*125     line                            
c                                                       ! scratch variable for plot output
      character*1       lonc
c							! east/west character
      integer		lond                           
c							! longitude degrees
      character*1       magcode
c							! type of magnitude
      real		misfit
c							! misfit value
      integer           ncfrm
c                                                       ! frame pen color
      integer           ncup
c                                                       ! up pen color
      integer           ncdwn
c                                                       ! down pen color
      integer           nsol
c							! number of fm obserations used in solution
      real              pi                              
c                                                       ! pi
      real		ratio
c							! % machine picks
      real              rlatm                           
c							! latitude minutes
      real              rlonm                           
c							! longitude minutes
      real              rmag
c							! magnitude
      real              rmax1                           
c                                                       ! radius of large circle
      real              rmax2                           
c                                                       ! radius of small circle
      character*(*)     title
c							! plot title
      real		sec
c							! second
      real		xpos
c                                                       ! x plot position
      real              xpos1                           
c                                                       ! leftmost x position of title, summary card, symbol legend
      real		ypos
c                                                       ! y plot position
      real              ypos1                           
c                                                       ! y plot position of title
      real              ypos2                           
c                                                       ! y plot position of symbol legend
      logical		www
c							! T(F) version is (not) for www page
	parameter (hunit = 8)
	xpos = xpos1
	ypos = ypos1
	read (event, 10, iostat = ios) 
     1 iyr, imo, ida, ihr, imin, sec, 
     2 latd, latc, rlatm, lond, lonc, rlonm, depth, rmag, magcode, 
     3 misfit, nsol, fit90, ratio, 
     4 iustrk, iudip, iurake
10	format (
     1 i4, 2i2, 1x, 2i2, f6.2, 
     2 i3, a1, f5.2, i4, a1, f5.2, f7.2, 2x, f5.2, t82, a1, 
     3 t96, f4.2, 1x, i3, 1x, f5.2, t116, f4.2, 1x, 
     4 t122, 3(i2, 1x))
	if (ios .ne. 0) then
	  print *, 'error reading format in pltnet: exiting'
	  stop
	endif
	if (latc .eq. ' ') latc = 'N'
	if (lonc .eq. ' ') lonc = 'W'
c
c output parametric info for html page
c
	if (www) then
	  open (hunit, file = 'fpplot.par', status = 'unknown')
	  line = 'USGS NORTHERN CALIFORNIA SEISMIC NETWORK'
	  write (hunit, '(a)') line
	  line = 'P FIRST-MOTION FOCAL MECHANISM'
	  write (hunit, '(a)') line
	  line = '(double-couple source assumed)'
	  write (hunit, '(a,/)') line
	  write (line, 20) imo, ida, iyr, ihr, imin, nint(sec)
20	  format ('Event Date & Time          : ', 2(i2.2, '/'),  
     1 i4, 1x, 2(i2.2, ':'), i2.2)
	  write (hunit, '(a)') line
	  write (line, 30) float(latd) + rlatm/60., latc,
     1 float(lond) + rlonm/60., lonc
30	  format ('Location                   : ', f7.4, 1x, a1, ', ', 
     1 f8.4, 1x, a1)
	  write (hunit, '(a)') line
	  write (line, 40) latd, rlatm, latc, lond, rlonm, lonc
40	  format ('                           : (', i2, ' deg. ', f5.2, 
     1 ' min. ', a1, ', ', i3, '.deg. ', f5.2, ' min. ', a1, ')')
	  write (hunit, '(a)') line
	  write (line, 45) depth, depth/1.6
45	  format ('Depth                      : ', f4.1, 
     1 ' km. deep (', f5.1, ' miles)')
	  write (hunit, '(a)') line
	  write (line, 70) rmag, magcode
70	  format ('Magnitude                  : ', f3.1, ' M', a1)
	  write (hunit, '(a)') line
	  write (line, 75) nsol
75	  format ('# P First motions          : ', i3)
	  write (hunit, '(a)') line
c	  write (line, 76), misfit, fit90 - misfit
c76	  format ('Solution misfit            : ',f4.2,' (+', f3.2, ')')
c	  write (hunit, '(a)') line
c	  write (line, 77) ratio
c77	  format ('Station distribution ratio : ', f4.2)
c	  write (hunit, '(a)') line
	  write (line, 80) iustrk
80	  format ('Strike uncertainty         : ', i3, ' deg.')
	  write (hunit, '(a)') line
	  write (line, 90) iudip
90	  format ('Dip    uncertainty         : ', i3, ' deg.') 
	  write (hunit, '(a)') line 
	  write (line, 100) iurake
100	  format ('Rake   uncertainty         : ', i3, ' deg.')
	  write (hunit, '(a)') line
	  close (hunit)
c
c	plot legend to the right
c
	  call plus (hite2, xpos1 + .0, ypos2 + hite2/2.)
	  call circle (hite2, 2.0*pi, xpos1 + .0, ypos2 + 2.5*hite2)
	  line = 'DOWN FIRST-MOTION (DILATATION)'
	  call symbol (xpos1 + .2, ypos2 + 2.*hite2, hite2, line,0., 
     1 len(line))
	  line = 'UP FIRST-MOTION (COMPRESSION)'
	  call symbol (xpos1 + .2, ypos2, hite2, line, 0., len(line))
	  line = 'BOLDFACE INDICATES TAKE-OFF' 
	  call symbol (xpos1+.0, ypos2 - 4.*hite2, hite2, line, 0., 
     1 len(line))
	  line = 'ANGLE > 90 DEGREES (0=DOWN)'
	  call symbol (xpos1+.0, ypos2 - 6.*hite2, hite2, line, 0., 
     1 len(line))
	  line = '3 NUMBERS NEXT TO FAULT PLANE'
	  call symbol (xpos1+.0, ypos2 - 10.*hite2, hite2, line, 0., 
     1 len(line))
	  line = 'INDICATE STRIKE, DIP, AND RAKE'
	  call symbol (xpos1+.0, ypos2 - 12.*hite2, hite2, line, 0., 
     1 len(line))
	  line = 'P & T 90%'
	  call symbol (cx2 - 0.3, cy2 - rmax2 - 0.25, hite2, line, 0., 
     1 len(line))
	  line = 'CONFIDENCE RANGE'
	  call symbol (cx2 - 0.55, cy2 - rmax2 - 0.4, hite2, line, 0., 
     1 len(line))

	else
c
c	plot titles, all parameters at top of page
c
	  if (ansc .eq. 'n') call newpen (2)
	  line = event(1:11)//':'//event(12:19)
	  if (event (131:131) .eq. '*') line = line(1:20)//' (MULTIPLE)'
	  call symbol (xpos, ypos, hite3, line, 0., 31)
	  line = event(21:38)
	  ypos = ypos - 0.20
	  call symbol (xpos, ypos, hite3, line, 0., 18)
	  line = 'DEPTH = '//event(41:45)//' KM'
	  ypos = ypos - 0.20
	  call symbol (xpos, ypos, hite3, line, 0., 16)
	  line = 'MAG = '//event(49:52)//' '//event(82:82)
	  ypos = ypos - 0.20
	  call symbol (xpos, ypos, hite3, line, 0., 12)
	  xpos = xpos1 + 3.0
	  ypos = ypos1
	  if (ansc .eq. 'n') call newpen (1)
	  line = 'RMS = '//event(66:69)//' S'
	  call symbol (xpos, ypos, hite3, line, 0., 12)
	  ypos =ypos - .20
	  line = 'DMIN = '//event(60:62)//' KM'
	  call symbol (xpos, ypos, hite3, line, 0., 13)
	  ypos =ypos - .20
	  line = 'AZM GAP = '//event(57:59)
	  call symbol (xpos, ypos, hite3, line, 0., 13)
	  ypos =ypos - .20
	  line = '# FM = '//event(101:103)
	  call symbol (xpos, ypos, hite3, line, 0., 10)
	  xpos = xpos1 + 5.0
	  ypos = ypos1
	  line = 'ERH = '//event(71:74)//' KM'
	  call symbol (xpos, ypos, hite3, line, 0., 13)
	  line = 'ERZ = '//event(76:79)//' KM'
	  ypos =ypos - .20
	  call symbol (xpos, ypos, hite3, line, 0., 13)
	  write (line, 1000), misfit, fit90 - misfit
1000	  format ('MISFIT = ', f4.2, ' (+', f3.2, ')')
	  ypos =ypos - .20
	  call symbol (xpos, ypos, hite3, line, 0., 20)
	  ypos =ypos - .20
	  line = 'STDR = '//event(111:114)
	  call symbol (xpos, ypos, hite3, line, 0., 11)
	  xpos = xpos1 + 7.0
	  ypos = ypos1
	  line = 'STRIKE UNCERTAINTY = '//event(121:123)
	  call symbol (xpos, ypos, hite3, line, 0., 24)
	  line = 'DIP UNCERTAINTY = '//event(125:126)
	  ypos =ypos - .20
	  call symbol (xpos, ypos, hite3, line, 0., 20)
	  line = 'RAKE UNCERTAINTY = '//event(127:129)
	  ypos =ypos - .20
	  call symbol (xpos, ypos, hite3, line, 0., 22)
	  write (line, '(a, i3)') '% MACHINE PICKS = ', 
     1 ifix(100.*ratio)
	  ypos = ypos - .20
	  call symbol (xpos, ypos, hite3, line, 0., 21)
	  call symbol (cx2 + 1.4, ypos2, hite2, title, 90., 80)
	  if (ansc .eq. 'n') then
	    call newpen (1)
	    line = 'UP'
	    xpos = xpos1
	    call symbol (xpos, ypos2 + 4.*hite2,hite2,line,0.,len(line))
	    line = 'DWN'
	    xpos = xpos + .2
	    call symbol(xpos,ypos2+4.*hite2,hite2,line,0.,len(line))
	    call newpen (4)
	    call plus (hite2, xpos1 + .1, ypos2 + hite2/2.)
	    call circle (hite2, 2.0*pi, xpos1 + .1, ypos2 + 2.5*hite2)
	    call newpen (1)
	    call plus (hite2, xpos1 + .3, ypos2 + hite2/2.)
	    call circle (hite2, 2.0*pi, xpos1 + .3, ypos2 + 2.5*hite2)
	    line(1:10) = 'DILATATION'
	    call symbol (xpos1 + .5, ypos2 + 2.*hite2,hite2,line,0.,10)
	    line(1:11) = 'COMPRESSION'
	    call symbol (xpos1 + .5, ypos2, hite2, line, 0., 11)
	  else
	    call newpen(ncup)
	    call plus (hite2, xpos1 + .3, ypos2 + hite2/2.)
	    call newpen(ncdwn)
	    call circle (hite2, 2.0*pi, xpos1 + .3, ypos2 + 2.5*hite2)
	    call newpen(ncfrm)
	    line(1:10) = 'DILATATION'
	    call symbol (xpos1 + .5, ypos2 + 2.*hite2, hite2,line,0.,10)
	    line(1:11) = 'COMPRESSION'
	    call symbol (xpos1 + .5, ypos2, hite2, line, 0., 11)
	    line = 'P & T 90% CONFIDENCE RANGE'
	    call symbol (cx2 - .9, ypos2, hite2, line, 0., len(line))
	  end if
	endif
c
c plot big & little stereo net perimeters
c
	call strnet (cx1, cy1, pi/180., rmax1)
	call strnet (cx2, cy2, pi/180., rmax2)
	return
	end
