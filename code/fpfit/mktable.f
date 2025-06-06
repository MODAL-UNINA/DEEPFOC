       program mktable
c
c     purpose:       create table from extented summary cards which have both fault planes plus p&t axes orientation
c
c     input file:    .fps file
c

c revisions: 3/15/2004 add output for b-axis
c revisions: 9/22/2004 fix occassional b-axis azimuth problem

      real              da1                             ! dip angle of first plane
      real              da2                             ! dip angle of second plane
      real              dd1                             ! dip direction of first plane
      real              dd2                             ! dip direction of second plane
      character*79      event                           ! part of hypo71 summary card
      character*50      filnam                          ! file name of data
      character*1	flag				! output option flag
      integer           i                               ! loop index
      integer           iday                            ! event day
      integer           ihr                             ! event hour
      integer           imo                             ! event month
      integer           isec                            ! event second
      integer		iunit				! input unit #
      integer           iyr                             ! event year
      character*140	line				! input line
      integer           min                             ! event minute
      integer           nobs                            ! number of first motion observations
      integer           num                             ! summary card counter
      integer		ounit				! output unit #
      real              pain                            ! angle of incidence of p axis
      real              paz                             ! azimuth of p axis
      real              pi                              ! pi
      real              rad                             ! pi/180
      real              sa1                             ! rake of first plane
      real              sa2                             ! rake of second plane
      real              sec                             ! event second remainder
      real              tain                            ! angle of incidence of t axis
      real              taz                             ! azimuth of t axis

      parameter (iunit = 2, ounit = 3)
      pi = atan(1.0)*4.0
      rad = pi/180.0
      tol = 1.0e-5
c
      filnam = 'data.fps'
10    call askc ('enter name of "fps" file', filnam)
      open (iunit, file = filnam, err = 10, status = 'old', blank =
     & 'zero')
      filnam = 'fps.table'
      call askc ('enter name of output file', filnam)
      open (ounit, file = filnam, status = 'unknown', blank =
     * 'zero')
      flag = 's'
15    call askc ('output downdip direction (d) or strike (s)?', flag)
      if (flag .ne. 's' .and. flag .ne. 'd') goto 15
      if (flag .eq. 's') then
        write (ounit, 20) ('-', i = 1, 144)
20      format('   # YEARMODA HRMN SECND LATITUDE LONGITUDE  DEPTH   ',
     1 'MAG  NS GAP  DMN  RMS  ERH  ERZ NFM      PLANE 1       PLANE 2',
     2 '     P-AXIS   T-AXIS   B-AXIS',
     2  /, t86, 
     3 '      STRK DIP RAK  STRK DIP RAK AZM PLNG AZM PLNG',
     3 ' AZM PLNG'
     4 , /, 144a1)
      else
        write (ounit, 25) ('-', i = 1, 144)
25      format('   # YEARMODA HRMN SECND LATITUDE LONGITUDE  DEPTH   ',
     1 'MAG  NS GAP  DMN  RMS  ERH  ERZ NFM      PLANE 1       PLANE 2'
     2 '     P-AXIS   T-AXIS   B-AXIS',
     2  /, t86, 
     3 '      DDR DIP  RAK  DDR DIP  RAK AZM PLNG AZM PLNG',
     3 ' AZM PLNG'
     4 , /, 144a1)
      endif
      num = 0
c
c read event 
c
60	if (num .lt. 9999) num = num + 1
        read (2, 70, end = 1000) event, dd1, da1, sa1, nobs
70      format (a79, 3x, f4.0, f3.0, f4.0, 7x,i3)
c
c compute nodal planes
c
        call auxpln (dd1, da1, sa1, dd2, da2, sa2)
c
c  compute "p" and "t" axes 
c
        call tandp (pain, tain, paz, taz, da1, da2, dd1, dd2, sa1, sa2,
     & pi, rad)
c
c  compute B axis, which is cross product of P and T axes 
c  compute the direction numbers of P and T
c
	pazr = (90. - paz)*rad
	tazr = (90. - taz)*rad
	pdpr = (90. - pain)*rad
	tdpr = (90. - tain)*rad
	pa1 = cos(pazr)*cos(pdpr)
	pa2 = sin(pazr)*cos(pdpr)
	pa3 = sin(pdpr)
	ta1 = cos(tazr)*cos(tdpr)
	ta2 = sin(tazr)*cos(tdpr)
	ta3 = sin(tdpr)
	ba1 = pa2*ta3 - pa3*ta2
	ba2 = pa3*ta1 - pa1*ta3
	ba3 = pa1*ta2 - pa2*ta1
	ptheta = 90 - atan(pa2/pa1)/rad
	ttheta = 90 - atan(ta2/ta1)/rad
	btheta = 90 - atan(ba2/ba1)/rad
	if (pa1 .lt. 0 .and. pa2 .lt. 0) ptheta = ptheta + 180.
	if (ta1 .lt. 0 .and. ta2 .lt. 0) ttheta = ttheta + 180.
	if (ba1 .lt. 0 .and. ba2 .lt. 0) btheta = btheta + 180.
	baz = btheta
	pdip = 90. -asin(pa3)/rad
	tdip = 90. -asin(ta3)/rad
	bain = 90. -asin(ba3)/rad

c sanity check. Form dot product between all vectors

	iflag = 0
90	rl1 = cos((90. - pain)*rad)
	xcos1 = rl1*sin((paz)*rad)
	ycos1 = rl1*cos((paz)*rad)
	zcos1 = sin((90.-pain)*rad)
	rl2 = cos((90. - tain)*rad)
	xcos2 = rl2*sin((taz)*rad)
	ycos2 = rl2*cos((taz)*rad)
	zcos2 = sin((90.-tain)*rad)
	rl3 = cos((90. - bain)*rad)
	xcos3 = rl3*sin((baz)*rad)
	ycos3 = rl3*cos((baz)*rad)
	zcos3 = sin((90.-bain)*rad)
	rl4 = cos((90. - bain)*rad)
	xcos4 = rl4*sin((baz - 180.)*rad)
	ycos4 = rl4*cos((baz - 180.)*rad)
	zcos4 = sin((90.-bain)*rad)
	if (xcos1*xcos2 + ycos1*ycos2 + zcos1*zcos2 .gt. tol) then
	  write (ounit, *) 'p-t dot product > ', tol
	  stop
	endif

c horrid kludge. I don't know what's wrong with the above cross-product
c calculations, but flipping the b-axis azimuth is the solution

	if ((xcos1*xcos3 + ycos1*ycos3 + zcos1*zcos3 .gt. tol) .or.
     1 (xcos2*xcos3 + ycos2*ycos3 + zcos2*zcos3 .gt. tol)) then
	  if ((xcos1*xcos4 + ycos1*ycos4 + zcos1*zcos4 .lt. tol) .and.
     1 (xcos2*xcos4 + ycos2*ycos4 + zcos2*zcos4 .lt. tol)) then
	    if (iflag .eq. 0) then
c	      write (ounit, *) 'p-b dot product > ', tol, 
c     1 '; flipping b-axis azimuth'
	      baz = btheta - 180.
	      goto 90
	    else

c flipping didn't help. Something is very wrong

	      write (ounit, *) 'p-b dot product > ', tol
	      stop
	    endif
	  endif
	endif
	  
c	write (77, *) num
c	write (77, *) 'p-t', xcos1*xcos2 + ycos1*ycos2 + zcos1*zcos2
c	write (77, *) 'p-b3', xcos1*xcos3 + ycos1*ycos3 + zcos1*zcos3
c	write (77, *) 't-b4', xcos2*xcos4 + ycos2*ycos4 + zcos2*zcos4
c	print *, baz, bain
c	print *, paz, ptheta, pain, pdip
c	print *, taz, ttheta, tain, tdip
	
c
c  write out table
c
	if (flag .eq. 's') then
	  dd1 = dd1 - 90.
	  if (dd1 .lt. 0.) dd1 = dd1 + 360.
          dd2 = dd2 - 90. 
          if (dd2 .lt. 0.) dd2 = dd2 + 360.
	end if
c	if (nobs .ne. 0) write (event(33:35), '(i3)') nobs
c        write(ounit, 80) num, iyr, imo, iday, ihr, min, isec, sec, 
c     & event, nint(dd1), nint(da1), nint(sa1), nint(dd2), 
c        write(ounit, 80) num, event, nobs
        write(ounit, 80) num, event, nobs,
     & nint(dd1), nint(da1), nint(sa1), nint(dd2), 
     & nint(da2), nint(sa2), nint(paz), nint(90. - pain), nint(taz), 
     & nint(90. - tain), nint(baz), nint(90. - bain)
80      format (i4, 1x, a79, 1x, i3, 1x, 2(2x, i4, i3, i5), 3(2x,i3,i4))
      goto 60
c
c end of file
c
1000  close (2)
      close (3)
      stop
      end
