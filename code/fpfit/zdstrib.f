	program zdstrib
c	zdstrib reads a summary card , prompts for the minimum permitted hypocentral separation
c	between events, and outputs a subset summary file satisfying separation criteria

	parameter (mxstat = 20000)

	real x(mxstat), y(mxstat), z(mxstat)
	character *164 sumcrd, line
	character *41 file
	logical y2k
	pio180 = 4.0*atan(1.0)/180.

	file = ' '
1	call askc('Enter file name of fps or hypoinverse summary cards',
     1  file)
	open (2, file = file, status = 'old', err = 1)
	ifor = 1
2	ifor = jask ('Enter format: 0=h71 or fps, 1=hypoinverse', ifor)
	if (ifor .ne. 0 .and. ifor .ne. 1) goto 2
c
c  is this y1k or y2k format?
c
        read (2, '(a)') line
	if (ifor .eq. 0) then
          if (line(15:15) .eq. '.' .and. line(34:34) .eq. '.') then
            y2k = .false.
          elseif (line(17:17) .eq. '.' .and. line(36:36) .eq. '.') then
            y2k = .true.
          else
            write (*, '(a)')
     1 'Unable to determine whether file is y1k or y2k format'
            stop    
          end if
	else
          if (line(9:9) .ne. ' ' .and.
     1 (line(17:17) .eq. ' ' .or. line(17:17) .eq. 's')) then
            y2k = .false.
          elseif (line(12:12) .ne. ' ' .and. line(17:17).ne.' ')then
            y2k = .true.
          else
          write (*, '(a)')
     1 'Unable to determine whether file is y1k or y2k format'
            stop
          endif
	endif
        rewind (2)
	file = 'zdstrib.out'
	call askc ('Enter name of output file', file)
	open (1, file = file, status = 'unknown')

	dmin = 1.
	dmin = askr ('Enter minimum event separation (km)', dmin)
	if (dmin .lt. 0.) dmin = 0.
	iz = 0
3	iz = jask ('Enter 0=epicentral, 1=hypocentral separation', iz)
	if (iz .ne. 0 .and. iz .ne. 1) goto 3
	if (ifor .eq. 0) then
	  ratio = 0.4 
	  ratio = askr ('enter minimum stdr ratio (0-1)', ratio)
	  if (ratio .lt. 0.) ratio = 0.
	  if (ratio .gt. 1.) ratio = 1.
	end if
	num = 1
10	read (2, 20, end = 100) sumcrd
20	format (a)
	ilen = leng(sumcrd)
	if (ifor .eq. 0) then
	  if (.not. y2k) then
	    read (sumcrd, 30) dlatdg, dlatmn, dlondg, dlonmn, depth,stdr
30	    format (18x, f2.0, 1x, f5.2, 1x, f3.0, 1x, f5.2, 2x, f5.2, 
     & t108, f5.1)
	  else
	    read (sumcrd, 31) dlatdg, dlatmn, dlondg, dlonmn, depth,stdr
31	    format (20x, f2.0, 1x, f5.2, 1x, f3.0, 1x, f5.2, 2x, f5.2, 
     & t110, f5.1)
	  endif
	  if (ratio .gt. 0. and. stdr .le. ratio) goto 10
	else
	  if (.not. y2k) then
	    read (sumcrd, 35) dlatdg, dlatmn, dlondg, dlonmn, depth
35	    format (t15, f2.0, 1x, f4.2, f3.0, 1x, f4.2, f5.2)
	  else
	    read (sumcrd, 36) dlatdg, dlatmn, dlondg, dlonmn, depth
36	    format (t17, f2.0, 1x, f4.2, f3.0, 1x, f4.2, f5.2)
	  endif
	endif
	if (iz .eq. 0) depth = 0.
	dlatdg = (dlatdg + dlatmn/60.)*pio180
	dlatdg = ggtogc(dlatdg)
	dlondg = -(dlondg + dlonmn/60.)*pio180
	if (num .eq. 1) call refpt(dlatdg, dlondg)
	call delaz (dlatdg, dlondg, delta, azm, bazm, xpos, ypos)
	if (num .gt. 1) then
	  do 40 i = 1, num - 1
	    dist = sqrt( (x(i) - xpos)**2 + (y(i) - ypos)**2 + (z(i) -
     & depth)**2)
	    if (dist .le. dmin) goto 10
40	  continue
	end if
	x(num) = xpos
	y(num) = ypos
	z(num) = depth
	write (1, '(a)') sumcrd(1:ilen)
	num = num + 1
	if (num .gt. mxstat) then
	  print *, '***** number of earthquakes exceeds ', mxstat,
     1 ' *****'
	  stop
	end if
	goto 10

100	close (2)
	close (1)
	stop
	end
