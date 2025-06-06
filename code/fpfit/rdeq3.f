      subroutine rdeq3 (ain, ainmin, ainmax, az, dist, distmx, eunit, 
     & event, fmagmn, iunit, kilsta, minobs, mxqual, mxstat,
     & nkil, nr, nrev, pobs, prcntx, prmk, resmax, revsta, sigmaf, stn,
     & sumwt, weight, wtobs, icmp, kdate, idate, evntid, netcode, src,
     & kilnet, revnet, dbg, macsrc, nmsrc, hndsrc, nhsrc, ncmpnt, okcmp,
     & cmpnt, loc, revloc, revchn)
c
c reads hypoinverse archive file. returns summary card and corresponding phase first motions, qualitites, angles of incidence,
c station names, and azimuths.  calculates standard deviation (sigmaf) of fit from estimated standard deviations of the data.
c the estimated data errors are control-file inputs; corresponding data weights are calculated in main and passed to this
c routine in the parameter "weight".
c
      integer           mxqual                          
c							! (input) maximum # of qualities permitted
      integer           mxstat                          
c							! (input) maximum # of stations permitted
      real              ain(mxstat)                     
c							! (output) ray incidence angles
      real              ainmax                          
c							! (input) maximum permitted angle of incidence
      real              ainmin                          
c							! (input) minimum permitted angle of incidence
      real              az(mxstat)                      
c							! (output) ray azimuth angles (corresponding to ain)
      logical		dbg
c							! (input) true(false) = do (not) issue warning messages
      real              dist(mxstat)                    
c							! (output) epicentral distance
      real              distmx                          
c							! (input) maximum permitted epicentral distance
      integer           eunit                           
c							! (input) logical unit # of output of error messages
      character*(*)     event                           
c							! (output) summary card
      integer           evntid                           
c                                                       ! event id #
      real              fmagmn                          
c							! (input) minimum permitted magnitude
      character*1	hndsrc(mxstat)
c							! (input) allowable hand-timed source codes
      integer           icmp                            
c							! (input) 1(0)=do (not) composite data into one mechanism; ievp on output
      integer           idate(mxstat,2)                 
c							! (input) date range of station reversal; 0=>open-ended
      integer           iunit                           
c							! (input) logical unit # of hypo71 listing file (input file)
      integer           kdate(mxstat,2)                 
c                                                       ! date range of kilsta; 0=>open-ended
      character*2	kilnet(mxstat)
c							! (input) seismic network code for kilsta
      character*5	kilsta(mxstat)                  
c							! (input) ignored station names
      character*1	macsrc(mxstat)
c							! (input) allowable machine source codes
      integer           minobs                          
c							! (input) minimum number of observations required
      character*2	netcode(mxstat)
c							! (input) seismic network code (if newfor)
      integer           ncmpnt
c							! (input) number of allowable component codes (okcmp)
      integer           nkil                            
c							! (input) number of ignored stations
      integer           nhsrc
c							! (input) number of allowed hand-timed source codes (hndsrc)
      integer           nmsrc
c							! (input) number of allowed machine source codes (macsrc)
      integer           nr                              
c							! (output) -1=eof, 0=skip event, nr>0 => number of stations
      integer           nrev                            
c							! (input) number of reversed stations
      character*3	okcmp(mxstat)
c							! (input) allowable component codes
      real              pobs(mxstat)                    
c							! (output) observed first motion polarities; .5=compression, -.5=dilatation
      real              prcntx                          
c							! (output) % of stations that are machine picked
      character*4       prmk(mxstat)                    
c							! (output) first motion description (eg. ipu0)
      real              resmax                          
c							! (input) maximum permitted p-residual
      character*3       revchn(mxstat)
c                                                       ! (input) channel code of reversed reversed station
      character*2       revloc(mxstat)
c                                                       ! (input) location code of reversed station
      character*2	revnet(mxstat)
c							! (input) seismic network code for revsta
      character*5       revsta(mxstat)                  
c							! (input) reversed station names
      real              sigmaf                          
c							! (output) calculated standard deviation of fit based on data errors
      character*(5)       stn(mxstat)                     
c							! (output) station names
      real              sumwt                           
c							! (output) sum of observed first motion weights
      real              weight(mxqual)                  
c							! (input) weights associated with qualities
      real              wtobs(mxstat)                   
c							! (input) observed first motions weights
      character*3	cmpnt(mxstat)
c							! (output) station component code
      character*2       loc(mxstat)                 
c                                                       ! (output) station location code
c
      real		amag1
c							! secondary magnitude 
      real		amag2
c							! secondary magnitude
      character*1	blast
c							! flag to indicate a blast
      character*1	bstflg
c							! magnitude designator for bstmag
      real		bstmag
c							! best magnitude
      character*1	cm(4)
c							! magnitude designator
      character*1	cm1
c							! secondary magnitude label on summary card
      character*1	cm2
c							! secondary magnitude label on summary card
      real              dep                             
c							! hypocenter depth (not used)
      real              dmin                            
c							! distance to nearest station (not used)
      real              erh                             
c							! horizontal error (not used)
      real              erz                             
c							! vertical error (not used)
      character*174     evline                          
c							! line for reading event summary
      character*1       e_w                          
c							! hemisphere for lat
      logical           first                           
c							! flag: t=first time into routine
      character*1       fm                              
c							! first motion direction (u, d, +, -)
      real              fmag                            
c							! event magnitude
      integer           i                               
c							! dummy loop index
      integer           ic                              
c							! number of characters in summary card
      integer           igap                            
c							! gap (not used)
      integer           ih                             
c							! origin hour 
      integer           im                             
c							! origin minute 
      integer           ios                             
c							! iostat error
      integer           ipwt                            
c							! qualiity assigned to p arrival
      integer           j                               
c							! dummy loop index
      integer           jdate                           
c							! date of event
      integer           jdate1                           
c							! year of event
      integer           jdate2                           
c							! month of event
      integer           jdate3                           
c							! day of event
      integer           jwt                             
c							! index for data weight
      integer           k                               
c							! counter of good phase readings
      integer           lat                             
c							! origin latitude (not used)
      character*120     line                            
c							! line of hypoinverse station data
      integer           llen                             
c							! line length
      integer           lon                             
c							! origin longitude (not used)
      integer           mpref(4)
c							! preference order of 4 magnitudes (USGS ML, UCB ML, coda, xmag)
      integer           nclas(20)                       
c							! number of observations in each data class
      logical		newfor
c							! flag: F/T=old/new hypoinverse station format
      character*1       n_s                          
c							! hemisphere for lon
      integer           nsp                             
c							! number of stations (not used)
      real              pres                            
c							! traveltime residual
      real              rms                             
c							! location rms (not used)
      real              sec                             
c							! origin second (not used)
      logical           shadow                           
c							! shadow card format found
      character*1       shdo                            
c							! shadow card
      character*1       src(mxstat)
c							! data source code
      character*4       stn4                          
c							! first 4 letters of station name
      character*1       stn5                          
c							! fifth letter of station name
      character*6	strg1
c							! scratch string
      character*6	strg2
c							! scratch string
      character*12	test_strg
c							! scratch string
      real              tmag(4)
c							! array of 4 potential magnitudes on summary card                            
      real              varf                            
c							! calculated variance in fit based on data errors.
      real              varf1                           
c							! summation of number of observations
      real              varf2                           
c							! summation of number of observations per class x corresponding weight
      real              wt                              
c							! weight assigned to p arrival
      real              xlat                            
c							! epicentral latitude (not used)
      real              xlon                            
c							! epicentral longitude (not used)
      real              xmag                            
c							! amplitude magnitude
      logical           y2k
c							! y2k format found for hypoinverse file                           
c
      data first/.true./
      data cm/'C', 'A', 'B', 'L'/
      data mpref/4, 3, 1, 2/
      save first, nclas, y2k, shadow, newfor
c
c reset values
c
	if (icmp .eq. 0 .or. (icmp .eq. 1 .and. first)) then
	  do 10 i = 1, mxqual
	    nclas(i) = 0
10	  continue
	  prcntx = 0.
	  sumwt = 0.
	endif
c
c what format is this?
c
c summary
c
	if (first) then
	  read (iunit, 30, end = 1000) evline
	  if (evline(9:9) .ne. ' ' .and. 
     1 (evline(17:17) .eq. ' ' .or. evline(17:17) .eq. 's')) then
	    y2k = .false.
	  elseif (evline(12:12) .ne. ' ' .and. evline(17:17).ne.' ')then
	    y2k = .true.
	  else
	    print *, 'rdeq2 error: y2k format is unclear'
	    stop
	  endif

c shadow

	  read (iunit, '(a)', end = 1000) evline
	  if (evline(1:1) .eq. '$') then
	    shadow = .true.
	  else
	    shadow = .false.
	  endif

c phase

	  if (shadow) read (iunit, '(a)', end = 1000) evline
	  llen = leng(evline)
	  if (.not. y2k) then
	    if (llen .lt. 95) then
	      newfor = .false.
	    else
	      newfor = .true.
	    endif
	  endif
	  rewind (iunit)
          first = .false.
        end if
        evline = ' '
c
c read summary card (skip non-summary cards)
c
20      read (iunit, 30, end = 1000) evline
30      format (a)
        if (shadow) read (iunit, 30, end = 1000) shdo
        ic = ichar (evline(1:1))
        if (ic .lt. 48 .or. ic .gt. 57) goto 20
	if (.not. y2k) then
          read (evline, 40) jdate1, jdate2, jdate3, xmag, fmag, cm1, 
     & amag1, cm2, amag2, blast, bstflg, bstmag
40	  format (3i2, t35, f2.1, t68, f2.1, t115, a1, f3.2, 3x, a1, 
     & f3.2, t77, a1, t139, a1, f3.2)
	  if (jdate1 .gt. 0 .and. jdate1 .lt. 67) then
	    jdate1 = 2000 + jdate1
	  else
	    jdate1 = 1900 + jdate1
	  endif
	else
          read (evline, 41) jdate1, jdate2, jdate3, xmag, fmag, cm1, 
     & amag1, cm2, amag2, blast, bstflg, bstmag
41	  format (i4, 2i2, t37, f3.2, t71, f3.2, t123, a1, f3.2, 3x, a1,
     & f3.2, t81, a1, t147, a1, f3.2)
	endif
        jdate = jdate1*10000 + jdate2*100 + jdate3
c c
c c Choose magnitude from preference list. Search down the list of mags in
c c the preferred order until a non-zero magnitude is found.
c 	tmag(3) = 0
c 	tmag(4) = 0
c c
c c Find the Berkeley & local mag if present
c c
c 	if (cm1 .eq. 'B') tmag(3) = amag1
c 	if (cm1 .eq. 'L') tmag(4) = amag1
c 	if (cm2 .eq. 'B') tmag(3) = amag2
c 	if (cm2 .eq. 'L') tmag(4) = amag2
c c
c c Assemble preference list
c c
c 	tmag(1) = fmag
c 	tmag(2) = xmag
c c
c c The preferred mag is the first non-zero one
c c
c 	do 45 i = 1,4
c 	  bstmag = tmag(mpref(i))
c 	  if (bstmag .gt. 0.) then
c 	    bstflg = cm(mpref(i))
c 	    goto 46
c 	  end if
c 45	continue
c c
c c All magnitudes are zero
c c
c	bstflg = ' '
46      if (bstmag .lt. fmagmn) then
          if (icmp .eq. 0) nr = 0
          return
        end if
c
c get the phase data
c
        if (icmp .eq. 0) then
          k = 1
        else
          k = nr + 1
        end if
50      stn(k) = '    '
        loc(k) = '--'
        read (iunit, 60, end = 70) line
60      format (a)
        if (shadow) read (iunit, 60,end=70) shdo
	if (.not. y2k) then
	  if (.not. newfor) then
	    read (line, '(a4)') stn4
	    stn5 = ' '
	    netcode(k) = '  '
	  else
             read (line, '(a4, t95, a1, 3x, a2)') stn4, stn5, netcode(k)
	  end if
	  stn(k) = stn4//stn5
	else
	  read (line, '(a5,a2)') stn(k), netcode(k)
	endif
c
c check for end of phase data
c
70      if (stn(k)(1:4) .eq. '    ') then
c
c end of event
c
	  if (blast .ne. ' ') then
c
c Skip blasts (since sources are non-double-couple
c
            if (dbg) write (eunit, 42) evline(1:14), blast
42	      format (/,' ', 'event: ', a14, 
     1 ' skipped: auxilliarly remark (=', a1, ') is not a blank', /)
              nr = 0
          elseif (k-1 .ge. minobs .or. (icmp.eq.1 .and. k.gt.1)) then
c
c reformat summary record into standard hypo71 summary format
c
            if (icmp .eq. 0 .or. (icmp .eq. 1 .and. nr .eq. 0)) then
	      if (.not. y2k) then
                read (evline, 75, iostat = ios) ih,im, sec, lat, n_s, 
     & xlat, lon, e_w, xlon, dep, nsp, igap, dmin, rms, erh, erz
75              format (6x,2i2, f4.2, i2, a1, f4.2, i3, a1, f4.2, f5.2,
     & 2x, i3, i3, f3.0, f4.2, 31x, 2f4.2)
	      else
                read (evline, 751, iostat = ios) ih,im, sec, lat, n_s, 
     & xlat, lon, e_w, xlon, dep, nsp, igap, dmin, rms, erh, erz
751             format (8x, 2i2, f4.2, i2, a1, f4.2, i3, a1, f4.2, f5.2,
     & t40, i3, i3, f3.0, f4.2, t86, 2f4.2)
	      endif
              write (event, 76) jdate1, jdate2, jdate3, ih,im, sec, lat, 
     1 n_s, xlat, lon, e_w, xlon, dep, bstmag, nsp, igap, dmin, rms, 
     2 erh, erz, bstflg
76            format (i4, 2i2.2, 1x,2i2.2, 1x, f5.2, i3, a1, f5.2,i4,a1,
     & f5.2, 2f7.2, i3, i4, f5.1, f5.2, 2f5.1, t82, a1)
            end if
            nr = k - 1
            prcntx = prcntx/float(nr)
            varf1 = 0.
            varf2 = 0.
            do 80 jwt = 1, mxqual
              varf1 = varf1 + nclas(jwt)
              varf2 = varf2 + nclas(jwt)*weight(jwt)
80          continue
            varf  = varf1/(varf2*varf2)
            sigmaf= sqrt(varf)
	    read (line, '(t63, i10)') evntid
          else if (icmp .eq. 0) then
            if (dbg) write (eunit, 85) evline(1:14), k - 1, minobs
85	    format (/,' ', 'event: ', a14, 
     1 ' skipped: # of valid readings (=', i4, ') <', i4,/)
            nr = 0
          end if
          return
        end if
c
c ignore this station?
c
        if (nkil .gt. 0) then
          do 90  i = 1, nkil
            if (stn(k)//netcode(k) .eq. kilsta(i)//kilnet(i) .and.
     & jdate .ge. kdate(i, 1) .and.
     & (kdate(i, 2) .eq. 0 .or. jdate .le. kdate(i, 2))) then
	      if (dbg)
     1 write (eunit, 105) stn(k)//netcode(k)//'   ', 
     2 'name is in "kil" list', evline(1:14)
	      goto 50
	    endif
90        continue
        end if
c
c  so far, so good: now check phase card for polarity, distance, quality
c
	if (.not. y2k) then
          read (line, 100) prmk(k), pres, dist(k), ain(k), az(k), 
     1 src(k), cmpnt(k)
100       format (4x, a4, t25, f4.2, t59, f4.1, f3.0, t76, f3.0, t92,
     1 a1, t96, a3)
	else
          read (line, 103) cmpnt(k), prmk(k), pres, dist(k), ain(k), 
     1 az(k), src(k), loc(k)
103       format (t10, a3, 1x, a4, t35, f4.2, t75, f4.1, f3.0, t92, 
     1 f3.0, t109, a1, 2x, a2)
	endif
	test_strg = stn(k)//netcode(k)//loc(k)//cmpnt(k)
c
c check for acceptable 3-letter component code
c
	if ((.not. y2k .and. newfor) .or. y2k) then
	  if (ncmpnt .gt. 0) then
	    do 101 i = 1, ncmpnt
	      if (cmpnt(k) .eq. okcmp(i)) goto 102
101	    continue
	  else
	    write (eunit, '(a)') 'no valid components set; see "chn"'
	    stop
	  endif
	  if (dbg) 
     1 write (eunit, 105) test_strg, 'invalid component', evline(1:14)
	  goto 50
	endif
102	call upstr (prmk(k), 4)
	read (prmk(k), '(2x, a1, i1)') fm, ipwt
        if (fm .ne. 'U' .and. fm .ne. 'D' .and. fm .ne. '+' .and.
     & fm .ne. '-' .and. fm .ne. 'C') then
	  if (dbg) 
     & write (eunit, 105) test_strg,'invalid first motion (='//fm//')', 
     & evline(1:14)
	  goto 50
	endif
        if (dist(k) .gt. distmx) then
	  if (dbg) then
	    write (strg1, '(f6.1)') dist(k)
	    write (strg2, '(f6.1)') distmx
	    write (eunit, 105) test_strg,
     &'epicentral distance (='//strg1//') > "dis" value (='//strg2//')', 
     & evline(1:14)
	  endif
	  goto 50
	endif
        if (ain(k) .lt. ainmin) then
	  if (dbg) then
	    write (strg1, '(f6.1)') ain(k)
	    write (strg2, '(f6.1)') ainmin
	    write (eunit, 105) test_strg, 
     & 'take-off angle (='//strg1//') < "ainmin" value (='//strg2//')', 
     & evline(1:14)
	  endif
	  goto 50
	endif
        if (ain(k) .gt. ainmax) then
	  if (dbg) then
	    write (strg1, '(f6.1)') ain(k)
	    write (strg2, '(f6.1)') ainmax
	    write (eunit, 105) test_strg, 
     & 'take-off angle (='//strg1//') > "ainmax" value (='//strg2//')', 
     & evline(1:14)
	  endif
	  goto 50
	endif
        if (abs(pres) .gt. resmax) then
	  if (dbg) then
	    write (strg1, '(f6.1)') abs(pres)
	    write (strg2, '(f6.1)') resmax
	    write (eunit, 105) test_strg, 
     & 'p-residual (='//strg1//') > "res" value (='//strg2//')', 
     & evline(1:14)
	  endif
	  goto 50
	endif
        if (dist(k) .eq. 0.) then
	  if (dbg) 
     & write (eunit, 105) test_strg, 'distance = 0', evline(1:14)
105       format (' ', a12, ' skipped: ', a, ' for event: ', a14)
          goto 50
        end if
c
c assign p-weight value based on hand or machine source
c
        if (ipwt .ge. mxqual/2) then
          wt = 0.
	  if (dbg) write (eunit, 105) test_strg,
     & 'p-wt='//prmk(k)(4:4), evline(1:14)
	  goto 50
	else
	  if (nmsrc .gt. 0) then
	    do 106 i = 1, nmsrc
	      if (src(k) .eq. macsrc(i)) then
                jwt = ipwt + mxqual/2 + 1
	        wt = weight(jwt)
                if (wt .ne. 0.) then
	          prcntx = prcntx + 1.
	          goto 108
	        elseif (dbg) then
	          write (eunit, 105) test_strg, 
     & 'assigned machine-pick error rate=0 for p-wt='//prmk(k)(4:4), 
     & evline(1:14)
	          goto 50
	        endif
	      endif
106	    continue
	  endif
	  if (nhsrc .gt. 0) then 
	    do 107 i = 1, nhsrc
	      if (src(k) .eq. hndsrc(i)) then
                jwt = ipwt + 1
                wt = weight(jwt)
                if (wt .ne. 0.) then
	          goto 108
	        else
	          if (dbg) write (eunit, 105) test_strg, 
     & 'assigned hand-pick error rate=0 for p-wt='//prmk(k)(4:4), 
     & evline(1:14)
	          goto 50
                end if
              end if
107	    continue
	  endif
	endif
        if (dbg) write (eunit, 105) test_strg, 
     & 'unknown data source (='//src(k)//')', 
     & evline(1:14)
	goto 50
c
c check for repeated phase card
c
108       if (k .gt. 2 .and. icmp .eq. 0) then
          do 120 j = 1, k - 1
            if (stn(k)//netcode(k) .eq. stn(j)//netcode(j)) then
	      if (dbg) 
     & write (eunit,105)test_strg,'multiple readings from same station',
     & evline(1:14)
              goto 50
            end if
120       continue
        end if
c
c flip polariites if station is designated as reversed
c
	if (nrev .gt. 0) then
          do 130 i = 1, nrev
            if (test_strg .eq.
     & revsta(i)//revnet(i)//revloc(i)//revchn(i) .and.
     & jdate .ge. idate(i, 1) .and.
     & (idate(i, 2) .eq. 0 .or. jdate .le. idate(i, 2))) then
              if (fm .eq. 'U') prmk(k)(3:3) = 'D'
              if (fm .eq. 'C') prmk(k)(3:3) = 'D'
              if (fm .eq. 'D') prmk(k)(3:3) = 'U'
              if (fm .eq. '+') prmk(k)(3:3) = '-'
              if (fm .eq. '-') prmk(k)(3:3) = '+'
              fm = prmk(k)(3:3)
	      if (dbg) write (eunit, 125) test_strg, evline(1:14)
125	      format (' ', 'polarity flipped for station: ', a12, 
     & ' for event: ', a14)
            end if
130       continue
        end if
c
        nclas(jwt) = nclas(jwt) + 1
        wtobs(k) = wt
        sumwt = sumwt + wt
        if (fm .eq. 'U' .or. fm .eq. '+' .or. fm .eq. 'C') then
          pobs(k) = .5
        else
          pobs(k) = -.5
        end if
c
c increment k and check number against array dimensions
c
        k = k + 1
        if (k .gt. mxstat) then
          write (eunit, *) 
     & '***** rdeq3 error: number of stations readings exceeds ', 
     & mxstat, 'for event: ', evline(1:14), ' *****'
          if (nr .gt. minobs) then
            nr = k - 1
            prcntx = prcntx/float(nr)
          else
            nr = 0
          end if
          return
        end if
c
c read another phase
c
        goto 50
c
c end of file
c
1000  nr = -1
      return
      end
