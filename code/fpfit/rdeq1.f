      subroutine rdeq1 (ain, ainmin, ainmax, az, dist, distmx, eunit, 
     & event, fmagmn, iunit, kilsta, minobs, mxqual, mxstat,
     & nkil, nr, nrev, pobs, prcntx, prmk, resmax, revsta, sigmaf, stn,
     & sumwt, weight, wtobs, icmp, kdate, idate, dbg)
c
c reads hypo71 output listing. returns summary card and corresponding phase first motions, qualitites, angles of incidence,
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
      logical           dbg
c                                                       ! (input) true(false) = do (not) issue warning messages
      real              dist(mxstat)                    
c							! (output) epicentral distance
      real              distmx                          
c							! (input) maximum permitted epicentral distance
      integer           eunit                           
c							! (input) logical unit # of output of error messages
      character*(*)     event                           
c							! (output) summary card
      real              fmagmn                          
c							! (input) minimum permitted magnitude
      integer           icmp                            
c							! (input) 1(0)=do (not) composite data into one mechanism; ievp on output
      integer           idate(mxstat,2)                 
c							! (input) date range of station reversal; 0=>open-ended
      integer           iunit                           
c							! (input) logical unit # of hypo71 listing file (input file)
      integer           kdate(mxstat,2)                 
c                                                       ! date range of kilsta; 0=>open-ended
      character*(*)     kilsta(mxstat)                  
c							! (input) ignored station names
      integer           minobs                          
c							! (input) minimum number of observations required
      integer           nkil                            
c							! (input) number of ignored stations
      integer           nr                              
c							! (output) -1=eof, 0=skip event, nr>0 => number of stations
      integer           nrev                            
c							! (input) number of reversed stations
      real              pobs(mxstat)                    
c							! (output) observed first motion polarities; .5=compression, -.5=dilatation
      real              prcntx                          
c							! (output) % of stations that are machine picked
      character*(*)     prmk(mxstat)                    
c							! (output) first motion description (eg. ipu0)
      real              resmax                          
c							! (input) maximum permitted angle of incidence
      character*(*)     revsta(mxstat)                  
c							! (input) reversed station names
      real              sigmaf                          
c							! (output) calculated standard deviation of fit based on data errors
      character*(*)     stn(mxstat)                     
c							! (output) station names
      real              sumwt                           
c							! (output) sum of observed first motion weights
      real              weight(mxqual)                  
c							! (input) weights associated with qualities
      real              wtobs(mxstat)                   
c							! (input) observed first motions weights
c
      character*80      evline                          
c							! temporary line for rearranging event format
      logical           first                           
c							! flag: t=first time into routine
      character*1       fm                              
c							! first motion direction (u, d, +, -)
      real              fmag                            
c							! event magnitude
      integer           i                               
c							! dummy loop index
      integer           ipwt                            
c							! qualiity assigned to p arrival
      integer           iyr
c							! 2-digit year
      integer           j                               
c							! dummy loop index
      integer           jdate                           
c							! date of event
      integer           jwt                             
c							! index for data weight
      integer           k                               
c							! counter of good phase readings
      character*80      line                            
c							! line of hypo71 station output
      character*1       m                               
c							! test for fortran carriage control
      integer           nclas(20)                       
c							! number of observations in each data class
      real              pres                            
c							! traveltime residual
      character*4       stn4
c                                                       ! first 4 letters of station name
      character*4       test                            
c							! 2nd-6th characters of line of hypo71 output
      real              varf                            
c							! calculated variance in fit based on data errors.
      real              varf1                           
c							! summation of number of observations
      real              varf2                           
c							! summation of number of observations per class x corresponding weight
      real              wt                              
c							! weight assigned to p arrival
c
      data first/.true./
      save first, nclas
c
c reset values
c
      if (icmp .eq. 0 .or. (icmp .eq. 1 .and. first)) then
        do 10 i = 1, mxqual
          nclas(i) = 0
10      continue
        prcntx = 0.
        sumwt = 0.
        first = .false.
      end if
c
c find line prior to summary card
c
20    read (iunit, 30, end = 1000) test
30    format (2x, a4)
      if (test .ne. 'DATE') goto 20
c
c read summary card
c
        read (iunit, 40, end = 1000) evline
40      format (1x, a80)
        read (evline, '(i6, 39x, f5.2)') jdate, fmag
	if (jdate .ge. 0 .and. jdate .lt. 670000) then
	  jdate = 20000000 + jdate
	else
	  jdate = 19000000 + jdate
	endif
c
c check magntitude
c
        if (fmag .lt. fmagmn) then
          if (icmp .eq. 0) nr = 0
          return
        end if
        read (iunit, 30, end = 1000) test
        read (iunit, 30, end = 1000) test
c
c check whether phase data or focal mechanism follow
c
        if (test .ne. 'STN ') goto 20
        if (icmp .eq. 0) then
          k = 1
        else
          k = nr + 1
        end if
50      stn(k) = '     '
        read (iunit, 60, end = 70) line
60      format (a)
        read (line, '(a1, a4)') m, stn4
	stn(k) = stn4//' '
c
c check for end of phase data
c
70      if (m.eq.'1' .or. stn(k) .eq. '     ' .or. 
     1 stn(k) .eq. 'DATE ') then
c
c end of event
c
          if (k - 1 .ge. minobs .or. (icmp .eq. 1 .and. k .gt. 1)) then
c
c reformat event into y2k hypo71 summary format
c
            if (icmp .eq. 0 .or. (icmp .eq. 1 .and. nr .eq. 0))  then
	      read (evline, '(i2)') iyr
	      if (iyr .ge. 0 .and. iyr .lt. 67) then
	        event = '20'//evline(1:53)//evline(57:60)//
     1 evline(54:56)//'.0'//evline(63:80)
	      else
	        event = '19'//evline(1:53)//evline(57:60)//
     1 evline(54:56)//'.0'//evline(63:80)
	      endif
	    endif
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
          else if (icmp .eq. 0) then
            if (dbg) write (eunit, 85) evline(1:14), k - 1, minobs
85          format (/, ' ', 'event: ', a14,
     1 ' skipped: # of valid readings (=', i4, ') <', i4,/)
            nr = 0
          end if
          return
        end if
c
c ignore this station?
c
        if (nkil .gt. 0) then
          do 90 i = 1, nkil
            if (stn(k) .eq. kilsta(i) .and. 
     1 jdate .ge. kdate(i, 1) .and.
     & (kdate(i, 2) .eq. 0 .or. jdate .le. kdate(i, 2))) then
              if (dbg)
     1 write (eunit, 105) stn(k),
     2 'name is in "kil" list', evline(1:14)
	      goto 50
	    endif
90        continue
        end if
c
c  so far, so good: now check phase card for polarity, distance, quality
c
        read (line, 100) dist(k), az(k), ain(k), prmk(k), pres
100     format (6x, f5.1, 1x, f3.0, 1x, f3.0, 1x, a4, t55, f5.2)
	call upstr(prmk(k), 4)
        read (prmk(k), '(2x, a1, i1)') fm, ipwt
        if (fm .ne. 'U' .and. fm .ne. 'D' .and. fm .ne. '+' .and.
     & fm .ne. '-' .and. fm .ne. 'C') then
          if (dbg)
     & write (eunit, 105) stn(k),'invalid first motion (='//fm//')',
     & evline(1:14)
	  goto 50
	endif
        if (dist(k) .gt. distmx) then
	  if (dbg)
     & write (eunit, 105) stn(k), 'epicentral distance > "dis" value',
     & evline(1:14)
	  goto 50
	endif
        if (abs(pres) .gt. resmax) then
	  if (dbg) write (eunit, 105) stn(k),
     & 'p-residual > "res" value', evline(1:14)
	  goto 50
	endif
        if (ain(k) .lt. ainmin .or. ain(k) .gt. ainmax) then
	  if (dbg)
     & write (eunit, 105) stn(k), 'take-off angle > "ain" value',
     & evline(1:14)
	  goto 50
	endif
        if (dist(k) .eq. 0.) then
          if (dbg) 
     & write (eunit, 105) stn(k), 'distance = 0',  evline(1:18)
105       format (' ', a5, ' skipped: ', a, ' for event: ', a18) 
          goto 50
        end if
        if (ipwt .ge. mxqual/2) then
          wt = 0.
	  if (dbg) write (eunit, 105) stn(k),
     & 'p-wt='//prmk(k)(4:4), evline(1:14)
          goto 50
        else
          jwt = ipwt + 1
          wt = weight(jwt)
          if (wt .eq. 0.) then
            if (dbg) write (eunit, 105) stn(k),
     & 'assigned hand-pick error rate=0 for p-wt='//prmk(k)(4:4),
     & evline(1:14)
	    goto 50
	  endif
	endif
c
c check for repeated phase card
c
        if (k .gt. 2 .and. icmp .eq. 0) then
          do 120 j = 1, k - 1
            if (stn(k) .eq. stn(j)) then
              if (dbg) write (eunit, 105) stn(k), 
     & 'multiple readings from same station', evline(1:18)
              goto 50
            end if
120       continue
        end if
c
c flip polariites if station is designated as reversed
c
        do 130 i = 1, nrev
          if (stn(k) .eq. revsta(i) .and. 
     1 jdate .ge. idate(i, 1) .and.
     2 (idate(i, 2) .eq. 0 .or. jdate .le. idate(i, 2))) then
            if (fm .eq. 'U') prmk(k)(3:3) = 'D'
            if (fm .eq. 'C') prmk(k)(3:3) = 'D'
            if (fm .eq. 'D') prmk(k)(3:3) = 'U'
            if (fm .eq. '+') prmk(k)(3:3) = '-'
            if (fm .eq. '-') prmk(k)(3:3) = '+'
            fm = prmk(k)(3:3)
	    if (dbg) write (eunit, 125) stn(k), evline(1:14)
125         format (' ', 'polarity flipped for station: ', a11,
     & ' for event: ', a14)
          end if
130     continue
c
        nclas(jwt) = nclas(jwt) + 1
        wtobs(k) = wt
        sumwt = sumwt + wt
        if (fm.eq.'U' .or. fm.eq.'+' .or. fm.eq.'C' .or. fm.eq.'c') then
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
     & '***** rdeq1 error: number of stations readings exceeds ', 
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
