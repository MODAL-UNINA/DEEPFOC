      subroutine input (eunit, funit, iunit, punit, sunit, lopen2,
     & lopen3, lopen4, ddelc, ddelf, del0c, distmx, dlamc, dlamf,
     & dphic, dphif, erate, fmagmn, iamwt, ibst, infmt, ires, irep,
     & kilsta, minobs, mxdip, mxqual, mxrake, mxstat, mxstrk, ndelc,
     & nkil, nlamc, nphic, nrev, phi0c, revsta, kdate, idate, irslam,
     & title, weight, xlam0c, ifin, ittl, icmp, resmax, ainmin, ainmax,
     & kilnet, revnet, dbg, ncmpnt, okcmp, macsrc, nmsrc, hndsrc, nhsrc,
     & revloc, revchn)

c user interface to program.  reads control parameters interactively or from file.  checks parameters and writes them to
c report file.  online help included

      integer           mxqual
c							! (input) maximum # of qualities permitted
      integer           mxstat
c							! (input) maximum # of stations permitted
      real              ainmax
c							! (output) maximum permitted angle of incidence
      real              ainmin
c							! (output) minimum permitted angle of incidence
      logical           dbg
c                                                       ! (output) true(false) = do (not) issue warning messages
      real              ddelc
c							! (output) fault dip increment in degrees for coarse search
      real              ddelf
c							! (output) fault dip increment in degrees for fine search
      real              del0c
c							! (output) initial fault dip angle in degrees for coarse search
      real              distmx
c							! (output) maximum permitted epicentral distance
      real              dlamc
c							! (output) fault rake increment in degrees for coarse search
      real              dlamf
c							! (output) fault rake increment in degrees for fine search
      real              dphic
c							! (output) fault strike increment in degrees for coarse search
      real              dphif
c							! (output) fault strike increment in degrees for fine search
      real              erate(mxqual)
c							! (output) assumed weighted error rates for data, read from control card
      integer           eunit
c							! (input) logical unit # of output of error messages
      real              fmagmn
c							! (output) minimum permitted magnitude
      integer           funit
c							! (input) logical unit # of output of fit listing for all strikes, dips
      character*(*)     hndsrc(mxstat)
c                                                       ! allowable hand-timed source codes
      integer           iamwt
c							! (output) flag controling amplitude weighting (0=omit, 1=use)
      integer           ibst
c							! (output) flag: 0(1)=do(not) calculate multiple solutions
      integer           icmp
c							! (output) flag: 1(0)=do (not) composite data into one mechanism
      integer           idate(mxstat,2)
c							! (output) date range of station reversal; 0=>open-ended
      integer           infmt
c							! 1=hypo71 print listing
c							! 2=hypoellipse
c							! 3=hypoinverse
      integer           irep
c							! (output) flag: 1(0)=do(not) report each fps to terminal when computed
      integer           ires
c							! (output) flag: 0(1)=(un)restricted search
      integer           ifin
c							! (output) flag: 1(0)=do (not) limit fine search to coarse search range
      integer           ittl
c							! (input) title option
      integer           iunit
c							! (input) logical unit # of hypo listing file
      integer           kdate(mxstat,2)
c                                                       ! date range of kilsta; 0=>open-ended
      character*(*)     kilnet(mxstat)
c                                                       ! (output) seismic network code for kilsta
      character*(*)     kilsta(mxstat)
c							! (output) ignored station names
      logical           lopen2
c							! (output) t if sunit open
      logical           lopen3
c							! (output) t if punit open
      logical           lopen4
c							! (output) t if funit open
      integer           minobs
c							! (output) minimum number of observations required
      character*(*)     macsrc(mxstat)
c                                                       ! (input) allowable machine source codes
      integer           mxdip
c							! (input) maximum # of dip increments permitted
      integer           mxrake
c							! (input) maximum # of rake increments permitted
      integer           mxstrk
c							! (input) maximum # of strike increments permitted
      integer           ncmpnt
c                                                       ! (input) number of allowable component codes (okcmp)
      integer           ndelc
c							! (output) number of fault dip increments for coarse search
      integer           nhsrc
c                                                       ! number of allowed hand-timed source codes (hndsrc)
      integer           nkil
c							! (output) number of ignored stations
      integer           nlamc
c							! (output) number of fault rake increments for coarse search
      integer           nmsrc
c                                                       ! number of allowed machine source codes (macsrc)
      integer           nphic
c							! (output) number of fault strike increments for coarse search
      integer           nrev
c							! (output) number of reversed stations
      character*3       okcmp(mxstat)
c                                                       ! (output) allowable component codes
      real              phi0c
c							! (output) initial fault strike angle in degrees for coarse search
      integer           punit
c							! (input) logical unit # of output of extended summary and ray parameters
      real              resmax
c							! (output) maximum permitted p-residual
      character*(*)     revchn(mxstat)
c                                                       ! (output) channel code of reversed reversed station
      character*(*)     revloc(mxstat)
c                                                       ! (output) location code of reversed station
      character*(*)     revnet(mxstat)
c                                                       ! (output) seismic network code for revsta
      character*(*)     revsta(mxstat)
c							! (output) reversed station names
      integer           sunit
c							! (input) logical unit # of output of extended summary cards
      character*(*)     title
c							! (output title
      real              weight(mxqual)
c							! (output) weights associated with qualities
      real              xlam0c
c							! (output) initial fault rake angle in degrees for coarse search

      logical           askl
c							! function
      real              askr
c							! function
      character*3     chncode
c                                                       ! channel code (infmt 3 only)
      character*3       cm
c							! command
      character*3       cmpnt
c                                                       ! component code
      integer           cunit
c							! logical unit # of control file
      character*24      datstr
c							! date
      character*8       date
c							! DATE_AND_TIME date
      character*10      time
c							! DATE_AND_TIME time
      real              ddlcdf
c							! default fault dip increment in degrees for coarse search
      real              ddlfdf
c							! default fault dip increment in degrees for fine search
      real              del1c
c							! maximum value of coarse search dip range in degrees
      real              dl0cdf
c							! default initial fault dip angle in degrees for coarse search
      real              dlmcdf
c							! default fault rake increment in degrees for fine search
      real              dlmfdf
c							! default fault rake increment in degrees for fine search
      real              dpdr0c
c							! minimum value of coarse search dip direction range in degrees
      real              dpdr1c
c							! maximum value of coarse search dip direction range in degrees
      real              dphcdf
c							! default fault strike increment in degrees for coarse search
      real              dphfdf
c							! default fault strike increment in degrees for fine search
      real              er
c							! summation of erate array
      integer           erflag
c							! error flag; non-zero indicates unable to open
      character*40      filnm1
c							! name of report output file; see eunit
      character*40      filnm2
c							! name of extended summary output file; see sunit
      character*40      filnm3
c							! name of solution output file; see punit
      character*40      filnm4
c							! name of fit function output file; see funit
      character*40      filnm5
c							! name of hypo input file; see iunit
      character*40      filnm6
c							! name of control file; see cunit
      character*40      filnmt
c							! full pathname of filnm5
      logical           first
c							! flag: t if first time into routine
      integer           i
c							! loop index
      integer           ichn
c							! array index
      integer           idy
c							! day
      integer           ik
c							! array index
      integer           imo
c							! month
      integer           inp
c							! control input unit number
      character*80      inst
c							! parameters on instruction line
      integer           ios
c							! iostat specifier
      integer           irsdel
c							! flag: (0)1=(no) restricted coarse search range for dip angle
      integer           irslam
c							! flag: (0)1=(no) restricted coarse search range for rake angle
      integer           irsphi
c							! flag: (0)1=(no) restricted coarse search range for strike angle
      integer           iyr
c							! year
      integer           jask
c							! function
      integer           leng
c							! function
      logical           linst
c							! t if no instruction on command line
      real              lm0cdf
c							! default initial fault rake angle in degrees for coarse search
      character*2       loccode
c                                                       ! location code (infmt 3 only)
      logical           lopen1
c							! t if eunit open
      logical           lopen5
c							! t if iunit open
      integer           ndlcdf
c							! default number of fault dip increments for coarse search
      character*2	ntcode
c							! network code (infmt 3 only)
      integer           nlmcdf
c							! default number of fault rake increments for coarse search
      integer           nphcdf
c							! default number of fault strike increments for coarse search
      logical           ok
c                                                       ! t if rev chncode is found in revchn list
      real              ph0cdf
c							! default initial fault strike angle in degrees for coarse search
      character*1       src
c							! data source variable
      character*5       statn
c							! station name
      real              wt
c							! summation of weight array
      real              xlam1c
c							! maximum value of coarse search rake range in degrees
      real              yescnt
c							! # of times prompt given without response
c
c set up default grid spacing
c
      parameter (ph0cdf =  0., dl0cdf = 10., lm0cdf = -180.)
      parameter (dphcdf = 20., ddlcdf = 20., dlmcdf =   20.)
      parameter (nphcdf =   9, ndlcdf =   5, nlmcdf =    18)
      parameter (dphfdf =  5., ddlfdf =  5., dlmfdf =   10.)

      data filnm1 /'none'/
      data filnm2 /'none'/
      data filnm3 /'none'/
      data filnm4 /'none'/
      data filnm5 /'none'/
      data filnm6 /'fpfit.inp'/
      data cunit /18/
      data inp/5/
      data irsphi/0/
      data irsdel/0/
      data ntcode/'  '/
      data first/.true./

      save first, cunit, filnm1, filnm2, filnm3, filnm4, filnm5,
     & filnm6, inp, dpdr0c, dpdr1c, del1c, xlam1c, ntcode

      if (first) then
c-------------------------------------------------------------------------------
c set default grid parameters
c-------------------------------------------------------------------------------
        dpdr0c = ph0cdf + 90.
        dpdr1c = dpdr0c + (nphcdf - 1)*dphcdf
        dphic = dphcdf
        dphif = dphfdf
        del0c = dl0cdf
        del1c = del0c + (ndlcdf - 1)*ddlcdf
        ddelc = ddlcdf
        ddelf = ddlfdf
        xlam0c = lm0cdf
        xlam1c = xlam0c + (nlmcdf - 1)*dlmcdf
        dlamc = dlmcdf
        dlamf = dlmfdf
c-------------------------------------------------------------------------------
c open & begin reading optional startup command file first time into routine
c-------------------------------------------------------------------------------
        write (*, 10)
10      format (' Fpfit uses 3-letter LOWER-CASE commands, which can be
     &followed by', /, ' parameters in free-format, or which display cur
     &rent values & generate prompts.', /, ' Type "hel" for information
     & on available commands.')
        first = .false.
	open (cunit, file=filnm6, form='formatted',status='old',err=100)
c	open (cunit, file=filnm6, form='formatted',status='old',err=100,	! VAX/VMS version
c     1 readonly)								! VAX/VMS version
        inp = cunit
        goto 100
      end if
c-------------------------------------------------------------------------------
c read commands from terminal after a prompt, or from a file
c-------------------------------------------------------------------------------
100   if (inp .eq. 5) write (6, 110)
110   format (' yes? ', $)
c-------------------------------------------------------------------------------
c read a command line
c-------------------------------------------------------------------------------
      if (inp .eq. 5) then
        read (inp, 120, iostat = ios) cm, inst
120     format (a3, a)
      else
        read (inp, 120, iostat = ios) cm, inst
      end if
c-------------------------------------------------------------------------------
c comment
c-------------------------------------------------------------------------------
      if (cm(1:1) .eq. '#') goto 100
c-------------------------------------------------------------------------------
c reformat "@" command to "jmp" command
c-------------------------------------------------------------------------------
      if (cm(1:1) .eq. '@') then
        inst = cm(2:3)//inst
        cm = 'jmp'
      end if
      linst=inst(1:10) .eq. '          '
c-------------------------------------------------------------------------------
c return to interactive mode at end of command line
c-------------------------------------------------------------------------------
      if (ios .eq. -1) then
        if (inp .eq. cunit) then
          close (cunit)
          inp = 5
        else
          yescnt = yescnt + 1
          if (yescnt .eq. 100) then
            print *, '***** program stopped; assumed runaway batch jo
     &b *****'
            stop
          end if
        end if
        goto 100
      end if
      yescnt = 0
c-------------------------------------------------------------------------------
c set up report output file
c-------------------------------------------------------------------------------
      if (cm .eq. 'out') then
        call readfl (inst, eunit, filnm1, 'new', erflag,
     & 'file for report output')
        if (erflag .ne. 0) then
          if (erflag .eq. 117) then
            print *, ' "out" file already exists - try again'
          else
            print *, ' error opening "out" file - try again'
          end if
        end if
c-------------------------------------------------------------------------------
c set up summary output file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'sum') then
        call readfl (inst, sunit, filnm2, 'new', erflag,
     & 'file for extended summary output')
        if (erflag .ne. 0) then
          if (erflag .eq. 117) then
            print *, ' "sum" file already exists - try again'
          else
            print *, ' error opening "sum" file - try again'
          end if
        else if (filnm2(1:4) .ne. 'none') then
          lopen2 = .true.
        else
          lopen2 = .false.
        end if
c-------------------------------------------------------------------------------
c set up polarity output file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'pol') then
        call readfl (inst, punit, filnm3, 'new', erflag,
     & 'file for solution output')
        if (erflag .ne. 0) then
          if (erflag .eq. 117) then
            print *, ' "pol" file already exists - try again'
          else
            print *, ' error opening "pol" file - try again'
          end if
        else if (filnm3(1:4) .ne. 'none') then
          lopen3 = .true.
        else
          lopen3 = .false.
        end if
c-------------------------------------------------------------------------------
c set up fit function output file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'fit') then
        call readfl (inst, funit, filnm4, 'new', erflag,
     & 'file for fit-function output')
        if (erflag .ne. 0) then
          if (erflag .eq. 117) then
            print *, ' "fit" file already exists - try again'
          else
            print *, ' error opening "fit" file - try again'
          end if
        else if (filnm4(1:4) .ne. 'none') then
          lopen4 = .true.
        else
          lopen4 = .false.
        end if
c-------------------------------------------------------------------------------
c set up hypocenter input file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'hyp') then
        call readfl (inst, iunit, filnm5, 'old', erflag,
     & 'file for hypocenter input')
        if(erflag.ne.0) print *, ' error opening "hyp" file - try again'
c-------------------------------------------------------------------------------
c read a save file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'jmp') then
        if (linst) then
          call askc ('command file to execute', filnm6)
        else if (filnm6(1:4) .ne. 'none') then
          filnm6 = inst(1:leng(inst))
        end if
	open (cunit,file=filnm6,status='old',iostat=ios)
c       open (cunit,file=filnm6,status='old',iostat=ios,readonly)		! VAX/VMS version
        if (ios .ne. 0) then
          print *, ' error opening "jmp" file - try again'
        else
          inp = cunit
        end if
c-------------------------------------------------------------------------------
c set distance cut-off
c-------------------------------------------------------------------------------
      else if (cm .eq. 'dis') then
        if (linst) then
          distmx = askr ('epicentral distance cut-off (km)', distmx)
        else
          read (inst, *, iostat = ios) distmx
        end if
        if (ios .ne. 0 .or. distmx .le. 0) then
          distmx = 99999.
          write (6, 180) cm
180       format (' *** error in "', a3, '" parameters - try again ***')
        end if
c-------------------------------------------------------------------------------
c set debug option
c-------------------------------------------------------------------------------
      else if (cm .eq. 'dbg') then
        if (linst) then
          dbg = askl ('output warning messages (T or F)?', dbg)
        else
          read (inst, *, iostat = ios) dbg
        end if
        if (ios .ne. 0) then
          dbg = .false.
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set angle of incidence range
c-------------------------------------------------------------------------------
      else if (cm .eq. 'ain') then
        if (linst) then
          ainmin = askr ('minimum angle-of-incidence (deg)', ainmin)
          ainmax = askr ('maximum angle-of-incidence (deg)', ainmax)
        else
          read (inst, *, iostat = ios) ainmin, ainmax
        end if
        if (ios .ne. 0 .or. ainmin .lt. 0 .or. ainmin .gt. 180. .or.
     & ainmax .lt. 0 .or. ainmax .gt. 180. .or. ainmax .le. ainmin) then
          ainmin = 0.
          ainmax = 180.
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set p-residual cutoff
c-------------------------------------------------------------------------------
      else if (cm .eq. 'res') then
        if (linst) then
          resmax = askr ('p-residual cutoff (sec)', resmax)
        else
          read (inst, *, iostat = ios) resmax
        end if
        if (ios .ne. 0 ) then
          resmax = 100.
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set minimum magnitude
c-------------------------------------------------------------------------------
      else if (cm .eq. 'mag') then
        if (linst) then
          fmagmn = askr ('minimum event magnitude', fmagmn)
        else
          read (inst, *, iostat = ios) fmagmn
        end if
        if (ios .ne. 0) then
          fmagmn = 0.
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set minimum # of first motion observations
c-------------------------------------------------------------------------------
      else if (cm .eq. 'obs') then
        if (linst) then
          minobs = jask ('minimum # of first-motion observations (>5)',
     & minobs)
        else
          read (inst, *, iostat = ios) minobs
        end if
        if (ios .ne. 0 .or. minobs .lt. 6) then
          minobs = 15
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set hypocenter input format
c-------------------------------------------------------------------------------
      else if (cm .eq. 'for') then
        if (linst) then
          print *, '1=hypo71 print listing'
          print *, '2=hypoellipse'
          print *, '3=hypoinverse archive'
          infmt = jask ('option', infmt)
        else
          read (inst, *, iostat = ios) infmt
        end if
        if (ios .ne. 0 .or. (infmt. lt. 1 .or. infmt .gt. 3)) then
          infmt = 1
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set flag controling amplitude weighting
c-------------------------------------------------------------------------------
      else if (cm .eq. 'amp') then
        if (linst) then
          print *, '0=omit amplitude weighting'
          print *, '1=weighted data by p-radiation amplitude'
          iamwt = jask ('option', iamwt)
        else
          read (inst, *, iostat = ios) iamwt
        end if
        if (ios .ne. 0 .or. (iamwt .ne. 0 .and. iamwt .ne. 1)) then
          iamwt = 0
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set flag controling calculation of multiple solutions
c-------------------------------------------------------------------------------
      else if (cm .eq. 'bst') then
        if (linst) then
          print *, '0=search for multiple solutions'
          print *, '1=output only best solution'
          ibst = jask ('option', ibst)
        else
          read (inst, *, iostat = ios) ibst
        end if
        if (ios .ne. 0 .or. (ibst .ne. 0 .and. ibst .ne. 1)) then
          ibst = 0
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set flag to montior execution progress at terminal
c-------------------------------------------------------------------------------
      else if (cm .eq. 'rep') then
        if (linst) then
          print *, '0=go about computation silently'
          print *, '1=report solutions to terminal when computed'
          irep = jask ('option', irep)
        else
          read (inst, *, iostat = ios) irep
        end if
        if (ios .ne. 0 .or. (irep .ne. 0 .and. irep .ne. 1)) then
          irep = 0
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set hand-picked error rates
c-------------------------------------------------------------------------------
      else if (cm .eq. 'hdr') then
        if (linst) then
          print *, 'assign error rates to p-wt code in percent/100.'
          print *, 'e.g., 0=0% first-motion error rate (ie., perfect dat
     &a)'
          print *, '      1=100% first-motion error rate (ie., always wr
     &ong)'
          do 190 i = 1, mxqual/2
            write (inst, '(a14, i1)') 'p=weight code ', i - 1
            erate(i) = askr (inst(1:15), erate(i))
190       continue
        else
          read (inst, *, iostat = ios) (erate(i), i = 1, mxqual/2)
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          do 200 i = 1, mxqual/2
            if (erate(i) .lt. 0. .or. erate(i) .gt. 1) write (6, 195)
     & 'invalid hand-timed error rate of ', erate(i), ' for code ', i -1
195         format ('0', a, e13.6, a, i1)
200       continue
        end if
c-------------------------------------------------------------------------------
c set machine-picked error rates
c-------------------------------------------------------------------------------
      else if (cm .eq. 'mcr') then
        if (linst) then
          print *, 'assign error rates to p-wt code in percent/100.'
          print *, 'e.g., 0=0% first-motion error rate (ie., perfect dat
     &a)'
          print *, '      1=100% first-motion error rate (ie., always wr
     &ong)'
          do 210 i = mxqual/2 + 1, mxqual
            write (inst, '(a14, i1)') 'p=weight code ', i - mxqual/2 - 1
            erate(i) = askr (inst(1:15), erate(i))
210       continue
        else
          read (inst,*,iostat= ios) (erate(i), i = mxqual/2 + 1, mxqual)
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          do 220 i = mxqual/2 + 1, mxqual
            if (erate(i) .lt. 0. .or. erate(i) .gt. 1) write (6, 195)
     & 'invalid machine-timed error rate of ', erate(i), ' for code ',
     & i - mxqual/2 - 1
220       continue
        end if
c-------------------------------------------------------------------------------
c set allowable hand sources
c-------------------------------------------------------------------------------
      else if (cm .eq. 'hds') then
        hndsrc(nhsrc + 1) = '-'
        if (linst) then
          src = '-'
          ichn = nhsrc + 1
          ichn = jask ('source # ', ichn)
          if (ichn .gt. 0) then
            if (ichn .gt. nhsrc) ichn = nhsrc + 1
            if (ichn .le. nhsrc) src = hndsrc(ichn)
            call askc ('1-letter "hand" data source code', src)
          end if
        else
          read (inst, *, iostat = ios) ichn, src
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          if (src .eq. '-' .and. (ichn .le. nhsrc)) then
            hndsrc(ichn) = '-'
          else if (src .ne. '-') then
            hndsrc(ichn) = src
            if (ichn .gt. nhsrc) nhsrc = ichn
          end if
        end if
c-------------------------------------------------------------------------------
c set allowable machine sources
c-------------------------------------------------------------------------------
      else if (cm .eq. 'mcs') then
        macsrc(nmsrc + 1) = '-'
        if (linst) then
          src = '-'
          ichn = nmsrc + 1
          ichn = jask ('source # ', ichn)
          if (ichn .gt. 0) then
            if (ichn .gt. nmsrc) ichn = nmsrc + 1
            if (ichn .le. nmsrc) src = macsrc(ichn)
            call askc ('1-letter "machine" data source code', src)
          end if
        else
          read (inst, *, iostat = ios) ichn, src
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          if (src .eq. '-' .and. (ichn .le. nmsrc)) then
            macsrc(ichn) = '-'
          else if (src .ne. '-') then
            macsrc(ichn) = src
            if (ichn .gt. nmsrc) nmsrc = ichn
          end if
        end if
c-------------------------------------------------------------------------------
c kill a station
c-------------------------------------------------------------------------------
      else if (cm .eq. 'kil') then
        kilsta(nkil + 1) = '-----'
        if (linst) then
          statn = 'none '
          ik = nkil + 1
          ik = jask ('station number', ik)
          if (ik .gt. 0) then
            if (ik .gt. nkil) ik = nkil + 1
            if (ik .le. nkil) statn = kilsta(ik)
            call askc ('(upto) 5-letter station name', statn)
            call askc
     & ('2-letter network code (hypoinverse only; otherwise blank)',
     & ntcode)
              kdate(ik, 1) = jask(
     1 'beginning date of kill (eg., 19880531; 0=open-ended',
     2 kdate(ik, 1))
              kdate(ik, 2) = jask(
     1 'ending date of kill (eg., 19880531;  0=open-ended',
     2 kdate(ik, 2))
          end if
        else
          read (inst, *, iostat = ios) ik, statn, ntcode, kdate(ik, 1),
     & kdate(ik, 2)
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          if (statn .eq. '-----' .and. (ik .le. nkil)) then
            kilsta(ik) = '-----'
          else if (statn .ne. '-----') then
            kilsta(ik) = statn
            kilnet(ik) = ntcode
            if (ik .gt. nkil) nkil = ik
            do 219 i = 1, 2
              if (kdate(ik, i) .ne. 0) then
                iyr = kdate(ik, i)/10000
                imo = kdate(ik, i)/100 - iyr*100
                idy = kdate(ik, i) - iyr*10000 - imo*100
                if ((imo .le. 0 .or. imo .gt. 12) .or.
     & (idy .le. 0 .or. idy .gt. 31)) then
                  if (i .eq. 1) then
                    write (6, 218) 'begin', statn
                  else
                    write (6, 218) 'end', statn
                  endif
 218              format (' invalid ', a, ' date on killed station "',
     & a, '"; try again')
                  kdate(ik, i) = 0
                end if
              end if
 219       continue
          end if
        end if
c-------------------------------------------------------------------------------
c reverse a station
c-------------------------------------------------------------------------------
      else if (cm .eq. 'rev') then
        revsta(nrev + 1) = '-----'
        if (linst) then
          statn = '-----'
          ik = nrev + 1
          ik = jask ('station number', ik)
          if (ik .gt. 0) then
            if (ik .gt. nrev) ik = nrev + 1
            if (ik .le. nrev) statn = revsta(ik)
            call askc ('(upto) 5-letter station name', statn)
            if (statn .ne. '-----') then
              if (infmt .eq. 3) then
                call askc ('2-letter network code)', ntcode)
                call askc ('2-letter location code (-- for none)',
     & loccode)
                call askc ('3-letter channel code', chncode)
              endif
              if (infmt .eq. 3 .and. ncmpnt .gt. 0) then
                ok = .false.
                do i = 1, ncmpnt
                    if (okcmp(i) .eq. chncode) ok = .true.
                end do
                if (.not. ok) write (6, 2201)
2201            format (' ', 'WARNING: input channel code not found in p
     &ermitted 3-letter channel codes (see chn command)')
              end if
              idate(ik, 1) = jask(
     1 'beginning date of reversal (eg., 19880531; 0=open-ended',
     2 idate(ik, 1))
              idate(ik, 2) = jask(
     1 'ending date of reversal (eg., 19880531;  0=open-ended',
     2 idate(ik, 2))
            end if
          end if
	else
          if (infmt .eq. 3) then
            read (inst, *, iostat = ios) ik, statn, ntcode, loccode,
     & chncode, idate(ik,1), idate(ik, 2)
          else
            read (inst, *, iostat = ios) ik, statn, ntcode, idate(ik,1),
     & idate(ik, 2)
          end if
	endif
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          if (statn .eq. '-----' .and. (ik .le. nrev)) then
            revsta(ik) = '-----'
          else if (statn .ne. '-----') then
            revsta(ik) = statn
            revnet(ik) = ntcode
            revloc(ik) = loccode
            revchn(ik) = chncode
            if (ik .gt. nrev) nrev = ik
            do 222 i = 1, 2
              if (idate(ik, i) .ne. 0) then
                iyr = idate(ik, i)/10000
                imo = idate(ik, i)/100 - iyr*100
                idy = idate(ik, i) - iyr*10000 - imo*100
                if ((imo .le. 0 .or. imo .gt. 12) .or.
     & (idy .le. 0 .or. idy .gt. 31)) then
                  if (i .eq. 1) then
                    write (6, 221) 'begin', statn
                  else
                    write (6, 221) 'end', statn
                  endif
221               format (' invalid ', a, ' date on reversed station "',
     & a, '"; try again')
                  idate(ik, i) = 0
                end if
              end if
222         continue
          end if
        end if
c-------------------------------------------------------------------------------
c set allowable component codes
c-------------------------------------------------------------------------------
      else if (cm .eq. 'chn') then
        okcmp(ncmpnt + 1) = '---'
        if (linst) then
          cmpnt = '---'
          ichn = ncmpnt + 1
          ichn = jask ('channel code number', ichn)
          if (ichn .gt. 0) then
            if (ichn .gt. ncmpnt) ichn = ncmpnt + 1
            if (ichn .le. ncmpnt) cmpnt = okcmp(ichn)
            call askc ('3-letter station channel code', cmpnt)
          end if
        else
          read (inst, *, iostat = ios) ichn, cmpnt
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
        else
          if (cmpnt .eq. '---' .and. (ichn .le. ncmpnt)) then
            okcmp(ichn) = '---'
          else if (cmpnt .ne. '---') then
            okcmp(ichn) = cmpnt
            if (ichn .gt. ncmpnt) ncmpnt = ichn
          end if
        end if
c-------------------------------------------------------------------------------
c save the current selection parameters in a file
c-------------------------------------------------------------------------------
      else if (cm .eq. 'sav') then
        if (linst .or. inst(1:4) .eq. 'e   ') then
          call askc ('file for save parameters', filnm6)
        else
          read (inst, '(a)') filnm6
        end if
        if (filnm6(1:4) .eq. 'none') then
          print *, 'no filename entered - try again'
          goto 100
        end if
	open (cunit,file=filnm6,form='formatted',status='unknown',
     1 iostat=erflag)
c       open (cunit, file = filnm6, form='formatted', status='new',		! VAX/VMS version
c    1 iostat = erflag, carriagecontrol = 'list')				! VAX/VMS version
        if (erflag .ne. 0) then
          print *, ' error opening "sav" file - try again'
          goto 100
        end if
        call params (mxqual, mxstat, ddelc, ddelf, del0c, del1c,
     & distmx, dlamc, dlamf, dpdr0c, dpdr1c, dphic, dphif, erate,
     & filnm1, filnm2, filnm3, filnm4, filnm5, filnm6, fmagmn, iamwt,
     & ifin, infmt, ittl, irep, kilsta, lopen2, lopen3, lopen4, minobs,
     & nkil, nrev, cunit, revsta, title, xlam0c, xlam1c, icmp, ainmin,
     & ainmax, resmax, ibst, kdate, idate, ncmpnt, okcmp, macsrc, nmsrc,
     & hndsrc, nhsrc, kilnet, revnet, revloc, revchn)
        close (cunit)
c-------------------------------------------------------------------------------
c show current selection parameters
c-------------------------------------------------------------------------------
      else if (cm .eq. 'sho') then
        call params (mxqual, mxstat, ddelc, ddelf, del0c, del1c,
     & distmx, dlamc, dlamf, dpdr0c, dpdr1c, dphic, dphif, erate,
     & filnm1, filnm2, filnm3, filnm4, filnm5, filnm6, fmagmn, iamwt,
     & ifin, infmt, ittl, irep, kilsta, lopen2, lopen3, lopen4, minobs,
     & nkil, nrev, 6, revsta, title, xlam0c, xlam1c, icmp, ainmin,
     & ainmax, resmax, ibst, kdate, idate, ncmpnt, okcmp, macsrc, nmsrc,
     & hndsrc, nhsrc, kilnet, revnet, revloc, revchn)
c-------------------------------------------------------------------------------
c set title
c-------------------------------------------------------------------------------
      else if (cm .eq. 'ttl') then
        if (linst) then
          print *, 'default title has form "hypo-file:date-of-computaton
     &"'
          print *, '0=no title'
          print *, '1=default title (hypo filename + date)'
          print *, '2=user-supplied title'
          ittl = jask ('option', ittl)
          if (ittl .eq. 0) then
            title = ' '
          else if (ittl .eq. 2) then
            call askc ('enter title (upto 80 char)', title)
          end if
        else
          read (inst, *, iostat = ios) ittl, title
        end if
        if (ios .ne. 0 .or. (ittl .lt. 0 .or. ittl .gt. 2)) then
          ittl = 1
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c restrict strike range
c-------------------------------------------------------------------------------
      else if (cm .eq. 'dir') then
        irsphi = 1
        if (linst) then
          print *, 'specify dip direction as downdip azimuth in degrees,
     & clockwise from north'
          call range ('minimum value of coarse search range', dpdr0c,
     & -180., 540.)
          call range ('maximum value of coarse search range', dpdr1c,
     & -180., 540.)
          call range ('increment in coarse search range', dphic, 1.,-1.)
          call range ('increment in fine search range', dphif, 1., -1.)
        else
          read (inst, *, iostat = ios) dpdr0c, dpdr1c, dphic, dphif
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
          irsphi = 0
        else
          if (dpdr1c .lt. dpdr0c) dpdr1c = dpdr1c + 360.
          nphic = int((dpdr1c - dpdr0c)/dphic) + 1
          phi0c = dpdr0c - 90.
          if (nphic .gt. mxstrk) then
            write (*, 300) mxstrk
300         format (/,' (total range)/(coarse increment) > array dimensi
     &on (=', i2, ')',/, ' either decrease range or increase coarse inte
     &rval (ie., try again)')
            irsphi = 0
          else if (nphic .le. 0) then
            write (*, 310)
310         format(/' (total range)/(coarse increment) <= 0; try again')
            irsphi = 0
          else if (inp .eq. 5) then
            print *, 'number of coarse strike increments =', nphic
          else if ((phi0c .eq. ph0cdf) .and. (dphic .eq. dphcdf) .and.
     & (dphif .eq. dphfdf) .and. (nphic .eq. nphcdf)) then
            irsphi = 0
          end if
        end if
c-------------------------------------------------------------------------------
c restrict dip range
c-------------------------------------------------------------------------------
      else if (cm .eq. 'dip') then
        irsdel = 1
        if (linst) then
          print *, 'specify dip angle down from horizontal in degrees'
          call range ('minimum value of coarse search range', del0c, 0.,
     & 180.)
          call range ('maximum value of coarse search range', del1c, 0.,
     & 180.)
          call range ('increment in coarse search range', ddelc, 1.,-1.)
          call range ('increment in fine search range', ddelf, 1., -1.)
        else
          read (inst, *, iostat = ios) del0c, del1c, ddelc, ddelf
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
          irsdel = 0
        else
          ndelc = int((del1c - del0c)/ddelc) + 1
          if (ndelc .gt. mxdip) then
            write (*, 300) mxdip
            irsdel = 0
          else if (ndelc .le. 0) then
            write (*, 310)
            irsdel = 0
          else if (inp .eq. 5) then
            print *, 'number of coarse dip increments =', ndelc
          else if ((del0c .eq. dl0cdf) .and. (ddelc .eq. ddlcdf) .and.
     & (ddelf .eq. ddlfdf) .and. (ndelc .eq. ndlcdf)) then
            irsdel = 0
          end if
        end if
c-------------------------------------------------------------------------------
c restrict rake range
c-------------------------------------------------------------------------------
      else if (cm .eq. 'rak') then
        irslam = 1
        if (linst) then
          print *, 'specify rake angle in degrees as follows:'
          print *, '0=left lateral, 90=reverse, -90=normal, +-180=right
     &lateral'
          call range ('minimum value of coarse search range', xlam0c,
     & -360., 360.)
          call range ('maximum value of coarse search range', xlam1c,
     & -360., 360.)
          call range ('increment in coarse search range', dlamc, 1.,-1.)
          call range ('increment in fine search range', dlamf, 1., -1.)
        else
          read (inst, *, iostat = ios) xlam0c, xlam1c, dlamc, dlamf
        end if
        if (ios .ne. 0) then
          write (6, 180) cm
          irslam = 0
        else
          nlamc = int((xlam1c - xlam0c)/dlamc) + 1
          if (nlamc .gt. mxrake) then
            write (*, 300) mxrake
            irslam = 0
          else if (nlamc .le. 0) then
            write (*, 310)
            irslam = 0
          else if (inp .eq. 5) then
            print *, 'number of coarse rake increments =', nlamc
          else if ((xlam0c .eq. lm0cdf) .and. (dlamc .eq. dlmcdf) .and.
     & (dlamf .eq. dlmfdf) .and. (nlamc .eq. nlmcdf)) then
            irslam = 0
          end if
        end if
c-------------------------------------------------------------------------------
c set flag controling whether fine search restricted to restricted coarse search range
c-------------------------------------------------------------------------------
      else if (cm .eq. 'fin') then
        if (linst) then
          print *, '0=fine search range not limited to restricted coarse
     & search range'
          print *, '1=fine search range limited to restricted coarse sea
     &rch range'
          ifin = jask ('option', ifin)
        else
          read (inst, *, iostat = ios) ifin
        end if
        if (ios .ne. 0 .or. (ifin .ne. 0 .and. ifin .ne. 1)) then
          ifin = 1
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c set flag for computing composite mechanisms
c-------------------------------------------------------------------------------
      else if (cm .eq. 'cmp') then
        if (linst) then
          print *, '0=compute separate mechanisms for each earthquake'
          print *, '1=compute a composite mechanism for all earthquakes
     &in file'
          icmp = jask ('option', icmp)
        else
          read (inst, *, iostat = ios) icmp
        end if
        if (ios .ne. 0 .or. (icmp .ne. 0 .and. icmp .ne. 1)) then
          icmp = 0
          write (6, 180) cm
        end if
c-------------------------------------------------------------------------------
c compute fault-plane solutions
c-------------------------------------------------------------------------------
      else if (cm .eq. 'fps') then
c
c check if required files are properly opened
c
        inquire (eunit, opened = lopen1)
        inquire (iunit, opened = lopen5, name = filnmt)
        if (.not. lopen1) then
          print *, 'error: report output file not open - type "out"'
          goto 100
        end if
        if (.not. lopen5) then
          print *,'error: hypocenter input file not open - type "hyp"'
          goto 100
        end if
        if (.not. (lopen2 .or. lopen3 .or. lopen4)) then
          print *,'error: no fault-plane solution output files opened'
          print *, '       type "sum", "pol", or "fit"'
          goto 100
        end if
c
c convert estimated error rates to weighting factors
c perfect error rates are tempered to a modest .001 to prevent infinite weights
c
        wt = 0.
        er = 0.
        do 225 i = 1, mxqual
          if (erate(i) .lt. 0.5) then
            if (erate(i) .lt. 0.001) then
              weight(i) = 29.6386
            else
              weight(i) = 1./sqrt(erate(i) - erate(i)*erate(i)) - 2.
            end if
          else
            weight(i) = 0.0
          end if
c
c check erates for likely errors
c
          wt = wt + weight(i)
          er = er + erate(i)
          if ((i .ne. 1 .and. i .ne. mxqual/2 + 1) .and.
     & (erate(i) .eq. 0.) .and. (erate(i) .lt. erate(i - 1))) then
            if (i .lt. mxqual/2 + 1) then
              write (6, 224) i - 1, 'hand-read', 'hdr'
            else
              write (6, 224) i - mxqual/2 - 1, 'machine-read', 'mcr'
            end if
224         format (/, ' error: the error rate for ', i1,
     & '-weight ', a, ' data is zero', /, '        type "', a, '"')
            goto 100
        end if
225     continue
        if (wt .eq. 0.) then
          print *, 'error: all error rates exceed 0.5'
          print *, '       type "hdr" or "mcr"'
          goto 100
        end if
        if (er .eq. 0.) then
          print *,'error: all error rates = 0 (an unrealistic estimate)'
          print *, '       type "hdr" or "mcr"'
          goto 100
        end if
        if (ittl .eq. 1) then
c
c get current time (using Fortran 90 INTRINSIC SUBROUTINE DATE_AND_TIME)
c
c         date is CCYYMMDD, time is hhmmss.sss
          Call DATE_AND_TIME( date, time )
c
c         format is MM/DD/YYYY hh:mm:ss.sss
          datstr = date(5:6) // '/' // date(7:8) // '/' // date(1:4) //
     1             ' ' //
     2             time(1:2) // ':' // time(3:4) // ':' // time(5:10)
          title = filnmt//'   '//datstr

        end if
        if (lopen3) write (punit, '(1x, a)') title
c
c if search range not specified, set to default
c
        if (irsphi .eq. 0) then
          phi0c = ph0cdf
          dphic = dphcdf
          dphif = dphfdf
          nphic = nphcdf
          dpdr0c = mod(phi0c + 90., 360.)
          dpdr1c = mod(dpdr0c + (nphic - 1)*dphic, 360.)
        else
          ires = 1
        end if
        if (irsdel .eq. 0) then
          del0c = dl0cdf
          ddelc = ddlcdf
          ddelf = ddlfdf
          ndelc = ndlcdf
          del1c = del0c + (ndelc - 1)*ddelc
        else
          ires = 1
        end if
        if (irslam .eq. 0) then
          xlam0c = lm0cdf
          dlamc = dlmcdf
          dlamf = dlmfdf
          nlamc = nlmcdf
          xlam1c = xlam0c + (nlamc - 1)*dlamc
        else
          ires = 1
        end if
        call params (mxqual, mxstat, ddelc, ddelf, del0c, del1c,
     & distmx, dlamc, dlamf, dpdr0c, dpdr1c, dphic, dphif, erate,
     & filnm1, filnm2, filnm3, filnm4, filnm5, filnm6, fmagmn, iamwt,
     & ifin, infmt, ittl, irep, kilsta, lopen2, lopen3, lopen4, minobs,
     & nkil, nrev, eunit, revsta, title, xlam0c, xlam1c, icmp, ainmin,
     & ainmax, resmax, ibst, kdate, idate, ncmpnt, okcmp, macsrc, nmsrc,
     & hndsrc, nhsrc, kilnet, revnet, revloc, revchn)
        write (eunit, *)
        if (inp .eq. cunit) then
          close (cunit)
          inp = 5
        end if
        return
c-------------------------------------------------------------------------------
c display a page of help text
c-------------------------------------------------------------------------------
      else if (cm .eq. 'hel' .or. cm .eq. '?  ') then
        write (6, 230)
230        format (/'  --- i/o commands ---'/
     & '  hyp - set file name of hypocenter input'/
     & '  for - set hypocenter input format'/
     & '  out - set file name of report output'/
     & '  sum - set file name of extended summary output'/
     & '  pol - set file name of solution and first-motion output'/
     & '  fit - set file name of fit-function output'/
     & '  ttl - set title'/
     & '  chn - set permitted 3-letter channel codes'/
     & '  dbg - report non-fatal warnings to "out" file (for=3 only)'/
     & '  rep - set option to monitor execution progress at terminal')
        write (6, 240)
240        format ('  --- do something ---'/
     & '  fps - compute to fault plane solutions'/
     & '  sho - display current command settings'/
     & '  sto - stop the program'/
     & '  jmp - execute a "sav" command file'/
     & '  sav - save current command settings in a file'/
     & '  @command_file - same as "jmp" command'/
     & '  #string - any line beginning with an "#" is considered a comme
     &nt line'//
     & '  --- for information on commands that control solution type "mo
     &r" ---'/)
      else if (cm .eq. 'mor') then
        write (6, 250)
250        format (//'  --- solution control ---'/
     & '  amp - set option to weight data by p-radiation amplitude funct
     &ion'/
     & '  bst - set option to search for multiple solutons'/
     & '  cmp - set option to generate a composite solution'/
     & '  mag - set minimum acceptable event magnitude'/
     & '  obs - set minimum # of p first-motions per event (ignored for
     &composites)'/
     & '  dis - set maximum permitable distance'/
     & '  res - set maximum permitable p-residual')
        write (6,260)
260        format ('  ain - set permitted angle-of-incidence range'/
     & '  kil - set names of stations to ignore'/
     & '  rev - set names of stations with reverse polarities'/
     & '  hdr*- assign first-motion error rates to p-wt codes of hand-pi
     &cked data'/
     & '  hds - set data source codes for hand-picked data'/
     & '  mcr - assign first-motion error rates to p-wt codes of machine
     &-picked data'/
     & '  mcs - set data source codes for machine-picked data'/
     & '  dir - set restricted downdip azimuth search range'/
     & '  dip - set restricted dip angle search range'/
     & '  rak - set restricted rake angle search range'/
     & '  fin - set option to restrict fine search range to coarse searc
     &h range')
c-------------------------------------------------------------------------------
c stop program
c-------------------------------------------------------------------------------
      else if (cm .eq. 'sto') then
        if (lopen2) close (sunit)
        if (lopen3) close (punit)
        if (lopen4) close (funit)
        close (eunit)
        close (iunit)
        stop
c-------------------------------------------------------------------------------
c i give up
c-------------------------------------------------------------------------------

      else if (cm .ne. '   ') then
        print *, cm, ' is an unknown command - try again'
      end if
      goto 100
      end
