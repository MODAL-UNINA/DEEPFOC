      subroutine params (mxqual, mxstat, ddelc, ddelf, del0c, del1c,
     & distmx, dlamc, dlamf, dpdr0c, dpdr1c, dphic, dphif, erate,
     & filnm1, filnm2, filnm3, filnm4, filnm5, filnm6, fmagmn, iamwt,
     & ifin, infmt, ittl, irep, kilsta, lopen2, lopen3, lopen4, minobs,
     & nkil, nrev, ounit, revsta, title, xlam0c, xlam1c, icmp, ainmin,
     & ainmax, resmax, ibst, kdate, idate, ncmpnt, okcmp, macsrc, nmsrc, 
     & hndsrc, nhsrc, kilnet, revnet, revloc, revchn)

c lists current parameter settings on unit ounit

      integer           mxqual                          
c							! maximum # of qualities permitted
      integer           mxstat                          
c							! maximum # of stations permitted
      real              ainmax                          
c							! maximum permitted angle of incidence
      real              ainmin                          
c							! minimum permitted angle of incidence
      real              ddelc                           
c							! fault dip increment in degrees for coarse search
      real              ddelf                           
c							! fault dip increment in degrees for fine search
      real              del0c                           
c							! minimum value of coarse search dip range in degrees
      real              del1c                           
c							! maximum value of coarse search dip range in degrees
      real              distmx                          
c							! maximum permitted epicentral distance
      real              dlamc                           
c							! fault rake increment in degrees for coarse search
      real              dlamf                           
c							! fault rake increment in degrees for fine search
      real              dpdr0c                          
c							! minimum value of coarse search dip direction range in degrees
      real              dpdr1c                          
c							! maximum value of coarse search dip direction range in degrees
      real              dphic                           
c							! fault strike increment in degrees for coarse search
      real              dphif                           
c							! fault strike increment in degrees for fine search
      real              erate(mxqual)                   
c							! assumed weighted error rates for data, read from control card
      character*40      filnm1                          
c							! name of report output file
      character*40      filnm2                          
c							! name of extended summary output file
      character*40      filnm3                          
c							! name of solution output file
      character*40      filnm4                          
c							! name of fit function output file
      character*40      filnm5                          
c							! name of hypo input file
      character*40      filnm6                          
c							! name of control file
      real              fmagmn                          
c							! minimum permitted magnitude
      character*(*)     hndsrc(mxstat)
c                                                       ! allowable hand-timed source codes
      integer           iamwt                           
c							! flag controling amplitude weighting (0=omit, 1=use)
      integer           ibst                            
c							! flag: 0(1)=do(not) calculate multiple solutions 
      integer           icmp                            
c							! flag: 1(0)=do (not) composite data into one mechanism
      integer           idate(mxstat,2)                 
c							! date range of station reversal; 0=>open-ended
      integer           ifin                            
c							! flag: 1(0)=do (not) limit fine search to coarse search range
      integer           infmt                           
c							! input file format : 1=hypo71, 2=hypoellipse, 3=hypoinverse
                                                        
c							!                              4=hypoinverse with shadow card
      integer           irep                            
c							! flag: 1(0)=do(not) report each fps to terminal when computed 
      integer           ittl                            
c							! title option: 0=no title
      integer           kdate(mxstat,2)                 
c							! date range of kilsta; 0=>open-ended
      character*(*)     kilnet(mxstat)
c                                                       ! seismic network code for kilsta
      character*(*)     kilsta(mxstat)                  
c							! ignored station names
      logical           lopen2                          
c							! t if sunit open
      logical           lopen3                          
c							! t if punit open
      logical           lopen4                          
c							! t if funit open
      character*(*)     macsrc(mxstat)
c                                                       ! allowable machine source codes
      integer           minobs                          
c							! minimum number of observations required
      integer           ncmpnt
c                                                       ! number of allowable component codes (okcmp)
      integer           nhsrc
c                                                       ! number of allowed hand-timed source codes (hndsrc)
      integer           nkil                            
c							! number of ignored stations
      integer           nmsrc
c                                                       ! number of allowed machine source codes (macsrc)
      integer           nrev                            
c							! number of reversed stations
      character*(*)     okcmp(mxstat)
c                                                       ! allowable component codes
      integer           ounit                           
c							! output unit #
      real              resmax                          
c							! maximum permitted p-residual
      character*3     revchn(mxstat)
c                                                       ! channel code of reversed reversed station
      character*2     revloc(mxstat)
c                                                       ! location code of reversed station
      character*2     revnet(mxstat)
c                                                       ! seismic network code for revsta
      character*5     revsta(mxstat)                  
c							! reversed station names
      character*(*)     title                           
c							! output title
      real              xlam0c                          
c							! minimum value of coarse search rake range in degrees
      real              xlam1c                          
c							! maximum value of coarse search rake range in degrees

      integer           i                              
c							! loop index
      integer           ifor1                          
c							! format statement label
      integer           ifor2                          
c							! format statement label
      integer           ifor3                          
c							! format statement label
      integer           leng                           
c							! function

c      if (ounit .eq. 6) then
c        assign 100 to ifor1
c        assign 200 to ifor2
c        assign 300 to ifor3
c        assign 400 to ifor4
c      else
c        assign 500 to ifor1
c        assign 600 to ifor2
c        assign 700 to ifor3
c        assign 800 to ifor4
c      end if
c
c if not writing out to save file, print a header
c
      if (ounit .eq. 8) then 
        write (ounit, '(a)') ' parameter settings for fpfit'
        write (ounit, '(a)') ' -----------------------------'
      end if
      if (ounit .eq. 6) then
        write (ounit, 200) 'ttl', ittl, title(1:leng(title))
        write (ounit, 100) 'hyp', filnm5(1:leng(filnm5))
        write (ounit, 100) 'out', filnm1(1:leng(filnm1))
        if (lopen2) write (ounit, 100) 'sum', filnm2(1:leng(filnm2))
        if (lopen3) write (ounit, 100) 'pol', filnm3(1:leng(filnm3))
        if (lopen4) write (ounit, 100) 'fit', filnm4(1:leng(filnm4))
        if (ounit .ne. 18) write (ounit, 100) 'jmp', 
     &filnm6(1:leng(filnm6))
        write (ounit, 200) 'for', infmt
        write (ounit, 300) 'mag', fmagmn
        write (ounit, 200) 'obs', minobs
        write (ounit, 300) 'dis', distmx
        write (ounit, 300) 'res', resmax
        write (ounit, 300) 'ain', ainmin, ainmax
        write (ounit, 200) 'amp', iamwt
        write (ounit, 200) 'bst', ibst
        write (ounit, 200) 'fin', ifin
        write (ounit, 200) 'rep', irep
        write (ounit, 200) 'cmp', icmp
        write (ounit, 300) 'hdr', (erate(i), i = 1, mxqual/2)
        write (ounit, 300) 'mcr', (erate(i), i = mxqual/2 + 1, mxqual)
        write (ounit, 300) 'dir', dpdr0c, dpdr1c, dphic, dphif
        write (ounit, 300) 'dip', del0c, del1c, ddelc, ddelf
        write (ounit, 300) 'rak', xlam0c, xlam1c, dlamc, dlamf
        if (ncmpnt .gt. 0) then
          do i = 1, ncmpnt
            write (ounit, 200) 'chn', i, okcmp(i)
          end do 
        endif
        if (nhsrc .gt. 0) then
          do i = 1, nhsrc
            write (ounit, 200) 'hds', i, hndsrc(i)
          end do 
        endif
        if (nmsrc .gt. 0) then
          do i = 1, nmsrc
            write (ounit, 200) 'mcs', i, macsrc(i)
          end do
        endif
      else
        write (ounit, 600) 'ttl', ittl, title(1:leng(title))
        write (ounit, 500) 'hyp', filnm5(1:leng(filnm5))
        write (ounit, 500) 'out', filnm1(1:leng(filnm1))
        if (lopen2) write (ounit, 500) 'sum', filnm2(1:leng(filnm2))
        if (lopen3) write (ounit, 500) 'pol', filnm3(1:leng(filnm3))
        if (lopen4) write (ounit, 500) 'fit', filnm4(1:leng(filnm4))
        if (ounit .ne. 18) write (ounit, 500) 'jmp', 
     &filnm6(1:leng(filnm6))
        write (ounit, 600) 'for', infmt
        write (ounit, 700) 'mag', fmagmn
        write (ounit, 600) 'obs', minobs
        write (ounit, 700) 'dis', distmx
        write (ounit, 700) 'res', resmax
        write (ounit, 700) 'ain', ainmin, ainmax
        write (ounit, 600) 'amp', iamwt
        write (ounit, 600) 'bst', ibst
        write (ounit, 600) 'fin', ifin
        write (ounit, 600) 'rep', irep
        write (ounit, 600) 'cmp', icmp
        write (ounit, 700) 'hdr', (erate(i), i = 1, mxqual/2)
        write (ounit, 700) 'mcr', (erate(i), i = mxqual/2 + 1, mxqual)
        write (ounit, 700) 'dir', dpdr0c, dpdr1c, dphic, dphif
        write (ounit, 700) 'dip', del0c, del1c, ddelc, ddelf
        write (ounit, 700) 'rak', xlam0c, xlam1c, dlamc, dlamf
        if (ncmpnt .gt. 0) then
          do i = 1, ncmpnt
            write (ounit, 600) 'chn', i, okcmp(i)
          end do 
        endif
        if (nhsrc .gt. 0) then
          do i = 1, nhsrc
            write (ounit, 600) 'hds', i, hndsrc(i)
          end do 
        endif
        if (nmsrc .gt. 0) then
          do i = 1, nmsrc
            write (ounit, 600) 'mcs', i, macsrc(i)
          end do
        endif
      endif
      if (ounit .eq. 6 .and. (nkil .gt. 0 .or. nrev .gt. 0)) then
        write (0, '(/, a)') ' hit carriage return to continue'
        read (5, '(i10)', err = 6) i
      end if
6     if (nkil .gt. 0) then
        do 10 i = 1, nkil
          if (ounit .eq. 6) then
            write (ounit, 200) 'kil', i, kilsta(i), kilnet(i),
     & kdate(i, 1), kdate(i, 2)
	  else
            write (ounit, 600) 'kil', i, kilsta(i), kilnet(i),
     & kdate(i, 1), kdate(i, 2)
	  endif
10      continue
      end if
      if (nrev .gt. 0) then
        do 20 i = 1, nrev
	  if (infmt .eq. 3) then
	    if (ounit .eq. 6) then
	      write (ounit, 400) 'rev', i, revsta(i), revnet(i),
     & revloc(i), revchn(i), idate(i, 1), idate(i, 2)
	    else
	      write (ounit, 800) 'rev', i, revsta(i), revnet(i),
     & revloc(i), revchn(i), idate(i, 1), idate(i, 2)
	    endif
	  else
	    if (ounit .eq. 6) then
              write (ounit, 200) 'rev', i, revsta(i), revnet(i), 
     & idate(i, 1), idate(i, 2)
	    else
              write (ounit, 600) 'rev', i, revsta(i), revnet(i), 
     & idate(i, 1), idate(i, 2)
	    endif
	  endif
20      continue
      end if
100   format (1x, a3, 1x, '''', a, '''')
200   format (1x, a3, 1x, i3, :, 1x, '''', a, '''', :, 1x, '''', a, 
     1 '''', :, 2(1x,i8))
300   format (1x, a3, 1x, 4g11.4)
400   format (1x, a3, 1x, i3, 1x, 4('''', a, '''', 1x), 2(1x,i8))
500   format (a3, 1x, '''', a, '''')
600   format (a3, 1x, i3, :, 1x, '''', a, '''', :, 1x, '''', a, 
     1 '''', :, 2(1x, i8))
700   format (a3, 1x, 4g11.4)
800   format (a3, 1x, i3, 1x, 4('''', a, '''', 1x), 2(1x,i8))
      return
      end
