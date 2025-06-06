      program fpfit
c
c
c     version 1.4  -  January 13, 1999
c
c     purpose:       Calculate double-couple fault plane solutions from p-wave first motions (see Reasenberg, P.  and
c                    D. Oppenheimer, Fpfit, Fpplot and Fppage: Fortran computer programs for calculating and displaying
c                    earthquake fault-plane solutions, U.S. Geological Survey Open-File Report 85-739).
c
c     required routines: all routines are enclosed
c
c      output:       1. an ascii file of y2k-hypo71 summary cards extended with fault plane solution parameters on logical unit sunit.
c                       this file serves as input to programs "qplot" and "plotpt".
c                    2. an ascii file consisting, for each earthquake, of the y2k-hypo71 extended summary card, followed by p & t axes 
c                       of neighboring solutions (within 90% confidence limits), followed by individual p-phase information,
c                       on logical unit punit. this file serves as input to programs "fpplot" and "fppage".
c                    3. an optional ascii file of the minimized fit function about the best solution on logical unit funit.
c                    4. an ascii file announcing the control file parameters, the presence of multiple mechanisms, errors in the
c                       data, a summary of polarity discrepancies by station and reading quality, and the distribution of strike, 
c                       dip, and rake uncertainties on logical unit eunit.
c
c      authors:      Paul Reasenberg and David Oppenheimer, U.S.G.S. in Menlo Park.  Some of the routines
c                    were adapted from code written by John Lahr, Bruce Julian, and Fred Klein.
c                    Mark Matthews, Stanford University, provided assistance in the error propagation analysis.
c
c      revisions in V1.4:
c		    1.	For Hypoinverse, code supports:
c		  	non-y2k and y2k versions of archive files
c			5-letter stations, 3 letter channel names, and netcode
c		    2.	Code output changed to y2k version of hypo71, which includes century.  
c		  	Assumes that years 67-99 are in 20th century, all other 2-digit years are assumed to be 21st century
c		    3.	Code now includes "dbg" option to that details reason for rejecting phases to "out" file
c
c Change 2007/06/01 Pete Lombard
c                   1.  For Hypoinverse Y2k version of archive files, read
C                       location code; and write channel, network and location
C                       codes to phase lines of pol file.
c                   2.  Changed the 'kil' command to require start and end
C                       dates, similar to the 'rev' command.
c
c	-----------------------------------------------------------------------
c	Although this program has been used by the USGS, no warranty, expressed
c	or implied, is made by the USGS or the United States Government as to
c	the accuracy and functioning of the program and related program
c	material nor shall the fact of distribution constitute any such
c	warranty, and no responsibility is assumed by the USGS in connection
c	therewith.
c	-----------------------------------------------------------------------
c	
c	
c	
c      installation notes: Check routines "readfl"  and "input" for VMS/SUN specific code
c
      integer           cunit                           
c                                                       ! logical unit # of input control file
      integer           eunit                           
c                                                       ! logical unit # of output of error messages
      integer           funit                           
c                                                       ! logical unit # of output of fit listing for all strikes, dips
      integer           iunit                           
c                                                       ! logical unit # of hypo71 listing file (input file)
      integer           mxdip                           
c                                                       ! maximum # of dip increments permitted
      integer           mxqual                          
c                                                       ! maximum # of qualities permitted
      integer           mxrake                          
c                                                       ! maximum # of rake increments permitted
      integer           mxslns                          
c                                                       ! maximum # of multiple solutions permitted
      integer           mxstat                          
c                                                       ! maximum # of stations permitted
      integer           mxstrk                          
c                                                       ! maximum # of strike increments permitted
      integer           punit                           
c                                                       ! logical unit # of output of extended summary and ray parameters
                                                        
c                                                       ! for use in plotting focal mech. diagrams with plotfm
      integer           sunit                           
c                                                       ! logical unit # of output of extended summary cards
c
      parameter (cunit = 1, eunit = 8, funit = 9, iunit = 2, mxdip = 19,
     & mxqual = 8, mxrake = 19, mxslns = 20, mxstat = 1200, mxstrk = 19,
     & punit = 3, sunit = 4)
c
      real              aerrc                           
c                                                       ! allowable angular difference between complimentary planes in coarse search
      real              aerrf                           
c                                                       ! allowable angular difference between complimentary planes in fine search
      real              ain(mxstat)                     
c                                                       ! ray incidence angles in degrees
      real              ainmax                          
c                                                       ! maximum permitted angle of incidence
      real              ainmin                          
c                                                       ! minimum permitted angle of incidence
      real              ainr                            
c                                                       ! ain converted to radians
      real              az(mxstat)                      
c                                                       ! ray azimuth angles (corresponding to ain)
      real              azr                             
c                                                       ! az converted to radians
      character*1       bdflag                          
c                                                       ! signals polarity discrepancy with best fit solution
      logical           best                            
c                                                       ! flag: true=best solution for event
      real              bot(mxdip,mxstrk,mxrake)        
c                                                       ! sum of product of observed and predicted weights
      character*2       cflag1                          
c                                                       ! signals minimum at edge of fine search grid
      character*1       cflag2                          
c                                                       ! signals minimum at edge of fine search grid
      real              coef(mxstat,6)                  
c                                                       ! coeficients by which moment tensor multiplied to give p radiation pattern
      character*3	cmpnt(mxstat)
c							! station component code
      logical           compl                           
c                                                       ! function
      real              da1                             
c                                                       ! dip angle of principle solution
      real              da2                             
c                                                       ! dip angle of auxilliary solution
      logical           dbg
c                                                       ! true(false) = do (not) issue warning messages
      real              dd1                             
c                                                       ! dip direction of principle solution
      real              dd2                             
c                                                       ! dip direction of auxilliary solution
      real              ddelc                           
c                                                       ! fault dip increment in degrees for coarse search
      real              ddelf                           
c                                                       ! fault dip increment in degrees for fine search
      real              del(mxdip)                      
c                                                       ! fault dip angle in degrees
      real              delc(mxdip)                     
c                                                       ! fault dip angle in degrees for coarse search
      real              delmax                          
c                                                       ! maximum dip range for solutions with fit<fitlim
      real              delmin                          
c                                                       ! minimum dip range for solutions with fit<fitlim
      real              del0c                           
c                                                       ! initial fault dip angle in degrees for coarse search
      real              del0f                           
c                                                       ! initial fault dip angle in degrees for fine search
      real              del1f                           
c                                                       ! terminating fault dip angle in degrees for fine search
      real              dip                             
c                                                       ! dip angle of best solution
      real              dist(mxstat)                    
c                                                       ! epicentral distance
      real              distmx                          
c                                                       ! maximum permitted epicentral distance
      real              dlamc                           
c                                                       ! fault rake increment in degrees for coarse search
      real              dlamf                           
c                                                       ! fault rake increment in degrees for fine search
      real              dphic                           
c                                                       ! fault strike increment in degrees for coarse search
      real              dphif                           
c                                                       ! fault strike increment in degrees for fine search
      real              eps
c                                                       ! machine precision
      real              epsp1
c                                                       ! machine precision plus 1
      real              erate(mxqual)                   
c                                                       ! assumed weighted error rates for each data class
      character*82      event                           
c                                                       ! summary card
      character*59      evfit                           
c                                                       ! dummy character string to hold fit values on output
      integer           evntid                           
c                                                       ! event id #
      character*50      filnam
c							! file name                          
      logical           first                           
c                                                       ! flag: true=first time into subroutine search
      real              fit90                           
c                                                       ! 90 % confidence limit of fit in fine search
      real              fit(mxdip,mxstrk,mxrake)        
c                                                       ! solution fit; weighted measure of agreement between obs, pred polarities
      real              fitlim                          
c                                                       ! 90% confidence upper limit of fitmnf, fitmnc
      real              fitmnc                          
c                                                       ! fit of best solution corresponding to fit(j1,n1,m1) of coarse search
      real              fitmnf                          
c                                                       ! fit of best solution corresponding to fit(j1,n1,m1) of fine search
      character*1       flag(mxdip,mxstrk,mxrake)       
c                                                       ! output string: if fit<fitlim then '*', otherwise blank
      character*1       flgc(mxslns)                    
c                                                       ! alpha equivalent of distinct solution #
      real              fmagmn                          
c                                                       ! minimum permitted magnitude
      real              gfit(mxdip*mxstrk*mxrake)       
c                                                       ! fits of "good" solutions found in coarse search
      character*1       hndsrc(mxstat)
c                                                       ! allowable hand-timed source codes
      integer           i                               
c                                                       ! loop index
      integer           iamwt                           
c                                                       ! code specifying type of amplitude weighting to use
      integer           ibst                            
c                                                       ! flag: 0(1)=do(not) calculate multiple solutions 
      integer           icmp                            
c                                                       ! flag: 1(0)=do (not) composite data into one mechanism
      integer           id                              
c                                                       ! loop index over ndst
      integer           idate(mxstat,2)                 
c                                                       ! date range of station reversal; 0=>open-ended
      integer           idip1                           
c                                                       ! dip angle of best fit
      integer           idst(mxslns,5)                  
c                                                       ! 1-3=indices of distinct solutions; 4=grid edge flag; 5=skip flag
      integer           idpdr1                          
c                                                       ! dip direction of best fit
      integer           idrng                           
c                                                       ! half-width dip range variation for solutions with fit<fitlim
      integer           ievp                            
c                                                       ! number of events processed
      integer           ievr                            
c                                                       ! number of events read
      integer           ifin                            
c                                                       ! flag: 1(0)=do (not) limit fine search to coarse search range
      integer           ifit(mxstrk)                    
c                                                       ! integer conversion of fit*100 for printer output
      integer           ig                              
c                                                       ! loop index over number of "good" solutions
      integer           igood(mxdip*mxstrk*mxrake,4)    
c                                                       ! array of indices of "good" solutions found in coarse search
      integer           ind(mxstat)                     
c                                                       ! pointer array to sorted order
      integer           index                           
c                                                       ! bin index into ndrng,nsrng,nrrng,nfit
      integer           indxa                           
c                                                       ! index of nearest p-,t-axis azimuth 
      integer           indxp                           
c                                                       ! index of nearest p-, t-axis plunge 
      integer           infmt                           
c                                                       ! input file format : 1=hypo71, 2=hypoinverse, 3=hypoellipse
                                                        
c                                                       !                     4=hypoinverse with shadow card
      integer           ipaxes(73,19)                   
c                                                       ! distinct coarse soltn # of p-axes 90% conf region (azm, plng in 5 deg inc)
      integer           ipwt                            
c                                                       ! weight assigned to p arrival
      integer           irep                            
c                                                       ! flag: 1(0)=do(not) report each fps to terminal when computed 
      integer           irpcnt                          
c                                                       ! counter for header output to terminal
      integer           ires                            
c                                                       ! flag: 0(1)=(un)restricted search
      integer           irrng                           
c                                                       ! half-width rake range variation for solutions with fit<fitlim
      integer           irslam                          
c                                                       ! flag: (0)1=(no) restricted coarse search range for rake angle 
      integer           isrc
c                                                       ! loop index
      integer           islip1                          
c                                                       ! rake of best fit
      integer           isrng                           
c                                                       ! half-width strike range variation for solutions with fit<fitlim
      integer           itaxes(73,19)                   
c                                                       ! distinct coarse soltn # of t-axes 90% conf region (azm, plng in 5 deg inc)
      integer           ittl                            
c                                                       ! title option
      integer           j                               
c                                                       ! loop index over dip
      integer           j1                              
c                                                       ! dip index of best solution
      integer           jmaxc                           
c                                                       ! largest dip index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           jmaxf                           
c                                                       ! largest dip index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           jminc                           
c                                                       ! smallest dip index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           jminf                           
c                                                       ! smallest dip index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           k                               
c                                                       ! loop index
      integer           kdate(mxstat,2)                 
c                                                       ! date range of kilsta 0=>open-ended
      character*2       kilnet(mxstat)
c                                                       ! seismic network code for kilsta
      character*5       kilsta(mxstat)                  
c                                                       ! ignored station names
      integer           l                               
c                                                       ! loop index over moment tensor
      character*2       loc(mxstat)                 
c                                                       ! station location code
      logical           lopen2                          
c                                                       ! t if sunit open
      logical           lopen3                          
c                                                       ! t if punit open
      logical           lopen4                          
c                                                       ! t if funit open
      integer           lpaxes(73,19)                   
c                                                       ! local solution p-axes 90% conf region (azm, plng in 5 deg inc)
      integer           ltaxes(73,19)                   
c                                                       ! local solution t-axes 90% conf region (azm, plng in 5 deg inc)
      character*124     line                            
c                                                       ! output string of nearby solutions orientations
      integer           m                               
c                                                       ! loop index over rake
      character*1       macsrc(mxstat)
c                                                       ! (input) allowable machine source codes
      integer           m1                              
c                                                       ! rake index of best solution
      integer           m1c                             
c                                                       ! m1 from coarse solution
      integer           minobs                          
c                                                       ! minimum number of observations required
      integer           mmax                            
c                                                       ! temporary value of mmaxc
      integer           mmaxc                           
c                                                       ! largest rake index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           mmaxf                           
c                                                       ! largest rake index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           mmin                            
c                                                       ! temporary value of mminc
      integer           mminc                           
c                                                       ! smallest rake index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           mminf                           
c                                                       ! smallest rake index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           n                               
c                                                       ! loop index of strike
      integer           naux                            
c                                                       ! solns index of complimentary solution
      integer           ncmpnt
c                                                       ! (input) number of allowable component codes (okcmp)
      integer           n1                              
c                                                       ! strike index of best solution
      integer           ndelc                           
c                                                       ! number of fault dip increments for coarse search
      integer           ndelf                           
c                                                       ! number of fault dip increments for fine search
      integer           ndlfdf                          
c                                                       ! default number of fault dip fine increments for unrestricted search 
      integer           ndrng(mxdip)                    
c                                                       ! number of dip solution ranges binned into ddelf degree increments
      integer           ndst                            
c                                                       ! number of distinct solutions found by hhog
      character*2       netcode(mxstat)
c                                                       ! seismic network code 
      integer           nfit(20)                        
c                                                       ! number of solutions binned into .025 fit increments
      integer           ng                              
c                                                       ! number of "good" solutions found in coarse search
      integer           nhsrc
c                                                       ! number of allowed hand-timed source codes (hndsrc)
      integer           nkil                            
c                                                       ! number of ignored stations
      integer           nlamc                           
c                                                       ! number of fault rake increments for coarse search
      integer           nlamf                           
c                                                       ! number of fault rake increments for fine search
      integer           nlmfdf                          
c                                                       ! default number of fault rake fine increments for unrestricted search 
      integer           nmaxc                           
c                                                       ! largest strike index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           nmaxf                           
c                                                       ! largest strike index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           nminc                           
c                                                       ! smallest strike index of coarse solution with fit<=fitlim about (j1,n1,m1)
      integer           nminf                           
c                                                       ! smallest strike index of fine solution with fit<=fitlim about (j1,n1,m1)
      integer           nmsrc
c                                                       ! number of allowed machine source codes (macsrc)
      integer           nphfdf                          
c                                                       ! default number of fault strike fine increments for unrestricted search 
      integer           nphic                           
c                                                       ! number of fault strike increments for coarse search
      integer           nphif                           
c                                                       ! number of fault strike increments for fine search
      integer           nr                              
c                                                       ! -1=eof, 0=skip event, nr>0 => number of stations
      integer           nrsv                            
c                                                       ! number of readings in a composite solution
      integer           nrev                            
c                                                       ! number of reversed stations
      integer           nrrng(mxrake + 1)               
c                                                       ! number of rake solution ranges binned into dlamf degree increments
      integer           nsol                            
c                                                       ! number of planes stored in array solns
      integer           nsrng(mxstrk)                   
c                                                       ! number of strike solution ranges binned into dphif degree increments
      integer           nstat                           
c                                                       ! total # of stations reporting for entire data set
      character*3       okcmp(mxstat)
c                                                       ! (input) allowable component codes
      real              pain                            
c                                                       ! angle of incidence of p axis (deg)
      real              paz                             
c                                                       ! azimuth of p axis (deg)
      real              phi(mxstrk)                     
c                                                       ! fault strike angle in degrees
      real              phic(mxstrk)                    
c                                                       ! fault strike angle in degrees for coarse search
      real              phimax                          
c                                                       ! maximum strike range for solutions with fit<fitlim
      real              phimin                          
c                                                       ! minimum strike range for solutions with fit<fitlim
      real              phi0c                           
c                                                       ! initial fault strike angle in degrees for coarse search
      real              phi0f                           
c                                                       ! initial fault strike angle in degrees for fine search
      real              phi1f                           
c                                                       ! terminating fault strike angle in degrees for fine search
      real              pi                              
c                                                       ! pi
      real              pobs(mxstat)                    
c                                                       ! observed first motion polarities; .5=compression, -.5=dilatation
      real              prad                            
c                                                       ! radiation amplitude corresponding ain, az.
                                                        
c                                                       ! (dilatation) -1.<prad<+1.(compression)
      real              prcntx                          
c                                                       ! % of stations that are machine picked
      character*4       prmk(mxstat)                    
c                                                       ! first motion description (eg. ipu0)
      integer           qcnt(mxqual,2)                  
c                                                       ! (input) indx 1=# of dscrpnt plrties for qlity, indx 2=# of observations
      real              qcntwt(mxqual,2)                
c                                                       ! index 1=weighted # dscrpnt polrities for quality, index 2=sum of weights
      real              rad                             
c                                                       ! conversion from degrees to radians
      real              resmax                          
c                                                       ! maximum permitted p-residual 
      character*2       revnet(mxstat)
c                                                       ! seismic network code for revsta
      character*5       revsta(mxstat)                  
c                                                       ! reversed station names
      real              sa1                             
c                                                       ! slip angle of principle solution
      real              sa2                             
c                                                       ! slip angle of auxilliary solution
      integer           scnt(mxstat,2)                  
c                                                       ! index 1=# of dscrpnt polarities for stat, index 2=# of observations
      real              scntwt(mxstat,2)                
c                                                       ! index 1=weighted # dscrpnt polarities for stat, index 2=sum of weights
      character*1       sflag                           
c                                                       ! flag indicating secondary solutions
      real              sigmaf                          
c                                                       ! calculated standard deviation of fit based on data errors
      real              slip                            
c                                                       ! slip angle of best solution
      real              solns(mxslns,3)                 
c                                                       ! array of final solutions used to check for redundancy
      character*1       src(mxstat)
c							! data source code
      character*5       stat(mxstat)                    
c                                                       ! names of all stations reporting
      real              stdr                            
c                                                       ! station distibution ratio
      character*5       stn(mxstat)                     
c                                                       ! station names per event
      character*5       string                          
c                                                       ! scratch variable
      real              strike                          
c                                                       ! strike of best solution
      real              sumwt                           
c                                                       ! sum of observed first motion weights
      real              tain                            
c                                                       ! angle of incidence of t axis (deg)
      real              taz                             
c                                                       ! azimuth of t axis (deg)
      character*80      title                           
c                                                       ! data set descriptor
      real              tm(6)                           
c                                                       ! moment tensor in upper triangular symetric storage mode
      real              u(3)                            
c                                                       ! cartesian p-wave direction vector (positive up, south, east)
      real              weight(mxqual)                  
c                                                       ! weights associated with qualities
      real              wtmax                           
c                                                       ! maximum weight
      real              wtobs(mxstat)                   
c                                                       ! observed first motions weights
      real              xlam(mxrake)                    
c                                                       ! fault rake angle in degrees
      real              xlamc(mxrake)                   
c                                                       ! fault rake angle in degrees for coarse search
      real              xlam0c                          
c                                                       ! initial fault rake angle in degrees for coarse search
      real              xlam0f                          
c                                                       ! initial fault rake angle in degrees for fine search
      real              xlam1f                          
c                                                       ! terminating fault rake angle in degrees for fine search
      real              xlmmax                          
c                                                       ! maximum rake range for solutions with fit<fitlim
      real              xlmmin                          
c                                                       ! minimum rake range for solutions with fit<fitlim
c
c  initialize statistics arrays to zero, control parameters to defaults
c
      data nstat, nfit /0, 20*0/
      data qcnt, qcntwt /mxqual*0, mxqual*0, mxqual*0.0, mxqual*0.0/
      data ndrng, nsrng, nrrng /mxdip*0, mxstrk*0, mxrake*0, 0/
      data flgc /'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
     & 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'/
      data erate /0.01, 0.02, 0.05, 0.1, 1.0, 1.0, 1.0, 1.0/
      data dbg /.false./
      data distmx /99999./
      data ainmin /0./
      data ainmax /180./
      data resmax /100./
      data fmagmn /0./
      data iamwt /0/
      data ibst /0/
      data irep /0/
      data irpcnt /19/
      data irslam /0/
      data icmp /0/
      data ifin /1/
      data nmsrc /1/
      data macsrc(1) /'W'/
      data nhsrc /1/
      data hndsrc(1) /'2'/
      data nkil /0/
      data nrev /0/
      data idate /mxstat*0, mxstat*0/
      data kdate /mxstat*0, mxstat*0/
      data infmt /3/
      data ittl /1/
      data title /'none'/
      data minobs /15/
      data ncmpnt /1/
      data okcmp(1) /'VHZ'/
      data ndlfdf /19/
      data nlmfdf /7/
      data nphfdf /19/
      data lopen2 /.false./
      data lopen3 /.false./
      data lopen4 /.false./

      pi = 4.*atan(1.0)
      rad = pi/180.
c
c calculate machine precision
c
      eps = 1.
5     eps = 0.5*eps
      epsp1 = eps + 1.
      if (epsp1 .gt. 1.) goto 5
      if (eps .eq. 0.) then
        print *, 'machine precision equals zero in main routine'
        stop
      end if
c
c get instructions 
c
10    call input (eunit, funit, iunit, punit, sunit, lopen2,
     & lopen3, lopen4, ddelc, ddelf, del0c, distmx, dlamc, dlamf,
     & dphic, dphif, erate, fmagmn, iamwt, ibst, infmt, ires, irep,
     & kilsta, minobs, mxdip, mxqual, mxrake, mxstat, mxstrk, ndelc,
     & nkil, nlamc, nphic, nrev, phi0c, revsta, kdate, idate, irslam,
     & title, weight, xlam0c, ifin, ittl, icmp, resmax, ainmin, ainmax, 
     & kilnet, revnet, dbg, ncmpnt, okcmp, macsrc, nmsrc, hndsrc, nhsrc)
c
      ievp = 0
      ievr = 0
      nr = 0
      nrsv = 0
      if (lopen4) call blurb (funit)
c
c find maximum weight
c
      wtmax = 0.
      do 20 i = 1, mxqual
        if (weight(i) .gt. wtmax) wtmax = weight(i)
20    continue
c
c read next event
c
30    ievr = ievr + 1
      if (infmt .eq. 1) then
        call rdeq1 (ain, ainmin, ainmax, az, dist, distmx, eunit, 
     & event, fmagmn, iunit, kilsta, minobs, mxqual, mxstat,
     & nkil, nr, nrev, pobs, prcntx, prmk, resmax, revsta, sigmaf, stn,
     & sumwt, weight, wtobs, icmp, kdate, idate, dbg)
      else if (infmt .eq. 2) then
        call rdeq2 (ain, ainmin, ainmax, az, dist, distmx, eunit, 
     & event, fmagmn, iunit, kilsta, minobs, mxqual, mxstat,
     & nkil, nr, nrev, pobs, prcntx, prmk, resmax, revsta, sigmaf, stn,
     & sumwt, weight, wtobs, icmp, kdate, idate, dbg)
      else if (infmt .ge. 3) then
       call rdeq3 (ain, ainmin, ainmax, az, dist, distmx, eunit,
     & event, fmagmn, iunit, kilsta, minobs, mxqual, mxstat,
     & nkil, nr, nrev, pobs, prcntx, prmk, resmax, revsta, sigmaf, stn,
     & sumwt, weight, wtobs, icmp, kdate, idate, evntid, netcode, src, 
     & kilnet, revnet, dbg, macsrc, nmsrc, hndsrc, nhsrc, ncmpnt, okcmp,
     & cmpnt, loc)
      end if
c
c continue accumulating data for composite until eof
c
        if (icmp .eq. 1) then
          if (nr .ge. 0) then
            if (nr .ne. nrsv) ievp = ievp + 1
            nrsv = nr
            goto 30
          else 
            nr = nrsv
            ievp = ievp - 1
            ievr = ievr - 1
            icmp = 0
            goto 40
          end if
        end if
c
c end of file
c
        if (nr .eq. -1) then
          close (iunit)
          ievr = ievr - 1
          if (nstat .gt. 0) then
            call fpout (ddelf, dlamf, dphif, erate, eunit, ievp, ievr,
     & ind, ires, mxdip, mxqual, mxrake, mxstat, mxstrk, ndelf, ndrng,
     & nfit, nlamf, nphif, nrev, nrrng, nsrng, nstat, qcnt, qcntwt,
     & revsta, scnt, scntwt, stat)
c
          else
            write (eunit, *) '***** fpfit error: no events satisfy in'//
     & 'put criteria *****'
            print *, '***** fpfit error: no events satisfy input criteri
     &a *****'
            inquire (eunit, name = filnam)
	    do 35 i = 50, 1, -1
	      if (filnam(i:i) .ne. ' ') goto 36
35	    continue
36	    print *, '      error messages can be found in ', filnam(1:i)
          end if
          goto 10
        end if
c
c insufficient readings skip event
c
        if (nr .eq. 0) goto 30
c
c good event: begin event loop
c set up p-wave direction unit vector  (up, south, east) for each ray
c
40      do 50 i = 1, nr
          ainr = ain(i)*rad
          azr = az(i)*rad
          u(1) = -cos(ainr)
          u(2) = -sin(ainr)*cos(azr)
          u(3) = sin(ainr)*sin(azr)
c
c find excitation coefficients for determining far-field p radiation pattern
c
          call pexcf (coef, i, mxstat, u)
50      continue
c 
c zero out the p-, t-axes 90% confidence region
c
        do 70 i = 1, 73
          do 60 j = 1, 19
            ipaxes(i,j) = 0
            itaxes(i,j) = 0
60        continue
70      continue
c
c coarse loop through fault models
c
        fit90 = 1.282 * sigmaf
        ievp = ievp + 1
        first = .true.
        call search (bot, coef, ddelc, del, delc, del0c, fit90, dlamc,
     & dphic, eps, first, fit, fitmnc, flag, gfit, iamwt, igood, j1,
     & m1, mxdip, mxrake, mxstat, mxstrk, n1, ndelc, ng, nlamc, nphic,
     & nr, phi, phic, phi0c, pobs, rad, wtobs, xlam, xlamc, xlam0c)
c
c determine distinct solutions from array of "good" solutions in coarse search
c
        call hhog (delc, eunit, j1, n1, m1, igood, ipaxes, itaxes, gfit,
     & ng, idst, ndst, mxdip, mxslns, mxstrk, mxrake, ndelc, nphic,
     & nlamc, phic, xlamc, bot, irslam)
        if (lopen4) then
c
c print out coarse fit function
c
          write (funit, 80) event
80        format (//, '***** coarse search *****', /, a)
          do 90 id = 1, ndst
            flag(idst(id, 1), idst(id, 2), idst(id, 3)) = flgc(id)
90        continue
          do 150 m = 1, nlamc
            write (funit, 100) ifix(xlamc(m))
100         format (///, 3x, 'slip angle = ', i4)
            write (funit, 110) (ifix(phic(n)), n = 1, nphic)
110         format (' strike:', 24(2x, i3))
            write (funit, *) ' dip'
            do 140 j = 1, ndelc
              do 120 n = 1, nphic
                if (lopen4) then
                  ifit(n) = ifix(1000.*fit(j, n, m))
                  if (ifit(n) .eq. 1000) ifit(n) = 999
                end if
120           continue
              write (funit, 130) ifix(delc(j)),
     & (ifit(n), flag(j, n, m), n = 1, nphic)
130           format (i4, 6x, 24(i3, a1, 1x), //)
140         continue
150      continue
        end if
        aerrf = 2.*max(ddelf, dlamf, dphif)
        aerrc = 2.*max(ddelc, dlamc, dphic)
        if (ibst .eq. 1) ndst = 1
        do 155 id = 1, ndst
          j1 = idst(id, 1)
          n1 = idst(id, 2)
          m1 = idst(id, 3)
          idst(id, 5) = 1
c
c  test whether minima is on edge of coarse seach grid.
c
          if (j1 .eq. 1 .or. j1 .eq. ndelc .or. n1 .eq. 1 .or. n1 .eq.
     & nphic .or. m1 .eq. 1 .or. m1 .eq. nlamc) then
            idst(id, 4) = 1
          else
            idst(id, 4) = 0
          end if
c
c express coarse solution in terms of dip direction, dip angle, and slip angle.
c
          call refrmt(delc(j1), idip1, idpdr1, islip1, phic(n1),
     & xlamc(m1))
          solns(id, 1) = idpdr1
          solns(id, 2) = idip1
          solns(id, 3) = islip1
c
c ascertain whether solution is an auxilliary plane of another solution in 
c coarse search. if so, set flag to skip fine search for solution which 
c resides at edge of coarse search grid, since these solutions are less likely
c to converge.
c
          if (id .gt. 1 .and. compl(solns, id - 1, float(idpdr1),
     & float(idip1), float(islip1), aerrc, mxslns, naux)) then
            if (idst(naux, 4) .eq. 1 .and. idst(id, 4) .eq. 0) then
              idst(naux, 5) = 0
            else 
              idst(id, 5) = 0
            end if
          end if
155     continue
c
c begin loop for fine search centered on each distinct solution found by hhog.
c skip searches on minima that have been identified as auxilliary planes of 
c other solutions.
c
        best = .true.
        nsol = 0
        do 430 id = 1, ndst
          if (idst(id, 5) .eq. 0) goto 430
          j1 = idst(id, 1)
          n1 = idst(id, 2)
          m1 = idst(id, 3)
          m1c = m1
          first = .false.
c
c  find fit90 range of coarse search
c
          fitlim = fitmnc + fit90
          jmaxc= 0
          jminc= ndelc
          nmaxc= 0
          nminc= nphic
          mmaxc= 0
          mminc= nlamc
          do 160 ig = 1, ng
            if (igood(ig, 4) .eq. id .and. gfit(ig) .le. fitlim) then
              j = igood(ig, 1)
              n = igood(ig, 2)
              m = igood(ig, 3)
              if (j .lt. jminc .and. n .eq. n1 .and. m .eq. m1) jminc =j
              if (j .gt. jmaxc .and. n .eq. n1 .and. m .eq. m1) jmaxc =j
              if (n .lt. nminc .and. j .eq. j1 .and. m .eq. m1) nminc =n
              if (n .gt. nmaxc .and. j .eq. j1 .and. m .eq. m1) nmaxc =n
              if (m .lt. mminc .and. j .eq. j1 .and. n .eq. n1) mminc =m
              if (m .gt. mmaxc .and. j .eq. j1 .and. n .eq. n1) mmaxc =m
            end if
160       continue
          if (mminc .eq. 1 .and. mmaxc .eq. nlamc) then
c
c check rake range for wrap around
c
            mmin = 1
            mmax = nlamc
            do 170 ig = 1, ng
              if (igood(ig, 4) .eq. id .and. gfit(ig) .le. fitlim .and.
     & igood(ig, 1) .eq. j1 .and. igood(ig, 2) .eq. n1) then
                if (igood(ig, 3) .eq. mmin + 1) mmin = mmin + 1
              end if
170         continue
            do 180 ig = ng, 1, -1
              if (igood(ig, 4) .eq. id .and. gfit(ig) .le. fitlim .and.
     & igood(ig, 1) .eq. j1 .and. igood(ig, 2) .eq. n1) then
                if (igood(ig, 3) .eq. mmax - 1) mmax = mmax - 1
              end if
180         continue
            if (mmin .ge. 1 .or. mmax .le. nlamc) then
c
c flag wrap around by making mminc > mmaxc
c
              mmaxc = mmin
              mminc = mmax
            end if
          end if
c
c determine fine search range for unrestricted search.
c restrict to fit90 range of coarse search if less than default fine search range
c
          if (ires .eq. 0 .or. (ires .eq. 1 .and. ifin .eq. 0)) then
            if (jminc .gt. 1) then
              del0f = max(delc(jminc - 1), delc(j1) -
     & float(ndlfdf/2)*ddelf)
            else
              del0f = delc(j1) - float(ndlfdf/2)*ddelf
            end if
            if (jmaxc .lt. ndelc) then
              del1f = min(delc(jmaxc + 1), delc(j1) +
     & float((ndlfdf + 1)/2)*ddelf)
            else
              del1f = delc(j1) + float((ndlfdf + 1)/2)*ddelf
            end if
            ndelf = min(ndlfdf, int((del1f - del0f)/ddelf) + 1)
c
            if (nminc .gt. 1) then
              phi0f = max(phic(nminc - 1), phic(n1) -
     & float(nphfdf/2)*dphif)
            else
              phi0f = phic(n1) - float(nphfdf/2)*dphif
            end if
            if (nmaxc .lt. nphic) then
              phi1f = min(phic(nmaxc + 1), phic(n1) +
     & float((nphfdf + 1)/2)*dphif)
            else
              phi1f = phic(n1) + float((nphfdf + 1)/2)*dphif
            end if
            nphif = min(nphfdf, int((phi1f - phi0f)/dphif) + 1)
c
c rake is different since it has wrap-around problems
c
            if (mminc .le. mmaxc) then
              xlam0f = max(xlamc(m1) - float(nlmfdf/2)*dlamf,
     & xlamc(mminc) - dlamc)
              xlam1f = min(xlamc(m1) + float((nlmfdf + 1)/2)*dlamf,
     & xlamc(mmaxc) + dlamc)
            else
              if (m1 .ge. mminc) then
                xlam0f = max(xlamc(m1) - float(nlmfdf/2)*dlamf,
     & xlamc(mminc) - dlamc)
                xlam1f = min(xlamc(m1) + float((nlmfdf + 1)/2)*dlamf,
     & xlamc(mmaxc) + 360. + dlamc)
              else
                xlam0f = max(xlamc(m1) - float(nlmfdf/2)*dlamf,
     & xlamc(mminc) - 360. - dlamc)
                xlam1f = min(xlamc(m1) + float((nlmfdf + 1)/2)*dlamf,
     & xlamc(mmaxc) + dlamc)
              end if
            end if
            nlamf = min(nlmfdf, int((xlam1f - xlam0f)/dlamf) + 1)
          else
c
c determine fine search for restricted search.  restrict to user-specified 
c restricted range as well as fit90 range of coarse search 
c
            del0f = max(delc(j1) - float(ndelc/2)*ddelc, del0c)
            if (jminc .gt. 1) del0f = max(del0f, delc(jminc - 1))
            del1f = min(delc(j1) + float((ndelc + 1)/2)*ddelc,
     & delc(ndelc))
            if (jmaxc .lt. ndelc) del1f = min(del1f, delc(jmaxc + 1))
            ndelf = min(mxdip, int((del1f - del0f)/ddelf) + 1)
c
            phi0f = max(phic(n1) - float(nphic/2)*dphic, phi0c)
            if (nminc .gt. 1) phi0f = max(phi0f, phic(nminc - 1))
            phi1f = min(phic(n1) + float((nphic + 1)/2)*dphic,
     & phic(nphic))
            if (nmaxc .lt. nphic) phi1f = min(phi1f, phic(nmaxc + 1))
            nphif = min(mxstrk, int((phi1f - phi0f)/dphif) + 1)
c
            xlam0f = max(xlamc(m1) - float(nlamc/2)*dlamc, xlam0c)
            if (mminc .gt. 1) xlam0f = max(xlam0f, xlamc(mminc - 1))
            xlam1f = min(xlamc(m1) + float((nlamc + 1)/2)*dlamc,
     & xlamc(nlamc))
            if (mmaxc .lt. nlamc) xlam1f = min(xlam1f, xlamc(mmaxc + 1))
            nlamf = min(mxrake, int((xlam1f - xlam0f)/dlamf) + 1)
          end if
c
c  do fine search
c
          call search (bot, coef, ddelf, del, delc, del0f, fit90, dlamf,
     & dphif, eps, first, fit, fitmnf, flag, gfit, iamwt, igood, j1,
     & m1, mxdip, mxrake, mxstat, mxstrk, n1, ndelf, ng, nlamf, nphif,
     & nr, phi, phic, phi0f, pobs, rad, wtobs, xlam, xlamc, xlam0f)
c
c express fine solution in terms of dip direction, dip angle, and slip angle.
c
          call refrmt(del(j1), idip1, idpdr1, islip1, phi(n1), xlam(m1))
c
c check again if fine search solution is an auxilliary plane of another solution
c or if it is the same as another solution in the list
c
          if (nsol .eq. 0 .or. (.not. compl(solns, nsol, float(idpdr1),
     & float(idip1), float(islip1), aerrf, mxslns, naux))) then
            nsol = nsol + 1
            solns(nsol, 1) = idpdr1
            solns(nsol, 2) = idip1
            solns(nsol, 3) = islip1
c            if (nsol .eq. 1) write (eunit, *) 'event ', event(1:19),
c     & ' has multiple solutions'
c 
c copy the p-, t-axes 90% confidence region for this discrete solution into 
c local array.
c
            do 200 i = 1, 73
              do 190 j = 1, 19
                if (ipaxes(i, j) .eq. id) then
                  lpaxes(i, j) = id
                else
                  lpaxes(i, j) = 0
                end if
                if (itaxes(i, j) .eq. id) then
                  ltaxes(i, j) = id
                else
                  ltaxes(i, j) = 0
                end if
190           continue
200         continue
c
c find the range of dip, strike and rake spanning each good solution for which 
c the fit is .le. fitlim.
c
            fitlim = fitmnf + fit90
            jmaxf= 0
            jminf= ndelf
            nmaxf= 0
            nminf= nphif
            mmaxf= 0
            mminf= nlamf
            do 210 m = 1, nlamf
              if (fit(j1, n1, m) .le. fitlim) then
                if (m .lt. mminf) mminf= m
                if (m .gt. mmaxf) mmaxf= m
              end if
210         continue
            do 220 n = 1, nphif
              if (fit(j1, n, m1) .le. fitlim) then
                if (n .lt. nminf) nminf= n
                if (n .gt. nmaxf) nmaxf= n
              end if
220         continue
            do 230 j = 1, ndelf
              if (fit(j, n1, m1) .le. fitlim) then
                if (j .lt. jminf) jminf= j
                if (j .gt. jmaxf) jmaxf= j
              end if
230         continue
c
c find largest dip, strike, rake half-width range for this solution
c
            delmin = min(delc(jminc), del(jminf))
            delmax = max(delc(jmaxc), del(jmaxf))
            phimin = min(phic(nminc), phi(nminf))
            phimax = max(phic(nmaxc), phi(nmaxf))
            if (mminc .le. mmaxc) then
              xlmmin = min(xlamc(mminc), xlam(mminf))
              xlmmax = max(xlamc(mmaxc), xlam(mmaxf))
            else if (m1c .ge. mminc) then
              xlmmin = min(xlamc(mminc), xlam(mminf))
              xlmmax = max(xlamc(mmaxc) + 360., xlam(mmaxf))
            else
              xlmmin = min(xlamc(mminc) -360., xlam(mminf))
              xlmmax = max(xlamc(mmaxc), xlam(mmaxf))
            end if
            idrng = nint((delmax - delmin)/2.)
            isrng = nint((phimax - phimin)/2.)
            irrng = nint((xlmmax - xlmmin)/2.)
c
c accumulate statistics on distribution of dip, strike, rake ranges for best 
c solutions only.
c
            if (best .and. (ires .eq. 0)) then
              index = idrng/5 + 1
              if (index .gt. mxdip) index = mxdip
              ndrng(index) = ndrng(index) + 1
              index = isrng/5 + 1
              if (index .gt. mxstrk) index = mxstrk
              nsrng(index) = nsrng(index) + 1
              index = irrng/10 + 1
              if (index .gt. mxrake + 1) index = mxrake
              nrrng(index) = nrrng(index) + 1
            end if
c
c  check for convergence (i.e., minima at grid edge)
c
            if ((ires .eq. 0) .and. (j1 .eq. 1 .or. j1 .eq. ndelf .or.
     & n1 .eq. 1 .or. n1 .eq. nphif .or. m1 .eq. 1 .or. m1 .eq. nlamf)) 
     & then
              cflag1 = 'no'
              cflag2 = 'C'
            else
              cflag1 = '  '
              cflag2 = ' '
            end if
            if (.not. best) then
              sflag = '*'
            else
              sflag = ' '
            end if
            stdr = bot(j1,n1,m1)/sumwt
c
c write out results
c
            if (infmt .eq. 3) then
              write (evfit, 240) idpdr1, idip1, islip1, fit(j1, n1, m1),
     &        nr, fitlim, stdr, prcntx, isrng, idrng, irrng, cflag2,
     &	      sflag, evntid
	    else
              write (evfit, 240) idpdr1, idip1, islip1, fit(j1, n1, m1),
     &        nr, fitlim, stdr, prcntx, isrng, idrng, irrng, cflag2, 
     &	      sflag
	    end if
240         format (i4, i3, i4, 2x, f4.2, 1x, i3, 1x, 2f5.2, 1x, f4.2,
     &      1x, 3i3, 2a1, i10)
            if (lopen4) write (funit, 250) event, evfit
250         format (/////, '***** fine search *****', /, a82, a59)
            if (lopen2) write (sunit, 260) event, evfit
260         format (a82, a59)
            if (lopen3) write (punit, 260) event, evfit
c            if (cflag2 .eq. 'C') write (eunit, 265) event(1:52), idpdr1,
c     & idip1, islip1
c265         format (1x, 'warning: event ', a52, i6, i5, i6,
c     & ' may not have converged to minima')
            if (irep .eq. 1) then
              if (irpcnt .eq. 19) then
                print *, '  #      ORIGIN TIME         LOCATION        '
     & //' DEPTH    MAG   DDR  DIP  RAKE CNVRG'
                print *, ' ---- ------------------- ------------------ '
     & //' -----    ---  ----- ---  ---- -----'
                irpcnt = 1
              end if
              if (best) then
                write (*, 270) ievp, event(1:52), idpdr1, idip1, islip1,
     & cflag1
270             format (1x, i4, 1x, a52, i6, i5, i6, 3x, a)
              else
                write (*, 280) idpdr1, idip1, islip1, cflag1
280             format (1x, '     MULTIPLE SOLUTION', t59, i6, i5, i6,
     & 3x, a)
              end if
              irpcnt = irpcnt + 1
            end if
c
c  loop over search area to print out fit parameters, 90% confidence region
c
            do 310 m = 1, nlamf
              if (lopen4) then
                write (funit, 100) ifix(xlam(m))
                write (funit, 110) (ifix(phi(n)), n = 1, nphif)
                write (funit, *) ' dip'
              end if
              do 300 j = 1, ndelf
                do 290 n = 1, nphif
                  if (lopen4) then
                    ifit(n) = ifix(1000.*fit(j, n, m))
                    if (ifit(n) .eq. 1000) ifit(n) = 999
                  end if
                  if (flag(j, n, m) .eq. '*') then
c
c find nearest grid cell in p, t axes arrays for 90% confidence region output 
c
                    da1 = del(j)
                    dd1 = phi(n) + 90.
                    sa1 = xlam(m)
                    call auxpln (dd1, da1, sa1, dd2, da2, sa2)
                    call tandp (pain, tain, paz, taz, da1, da2, dd1,
     & dd2, sa1, sa2, pi, rad)
                    indxa = nint(paz/5.) + 1
                    indxp = nint(pain/5.) + 1
                    lpaxes(indxa, indxp) = id
                    indxa = nint(taz/5.) + 1
                    indxp = nint(tain/5.) + 1
                    ltaxes(indxa, indxp) = id
                  end if
290             continue
                if (lopen4) write (funit, 130) ifix(del(j)),
     & (ifit(n), flag(j, n, m), n = 1, nphif)
300           continue
310         continue
c
c  output p-axes confidence region
c
            if (lopen3) then
              k = 0
              do 340 i = 1, 73
                do 330 j = 1, 19
                  if (lpaxes(i, j) .eq. id) then
                    k = k + 1
                    write (string, '(i3, i2)') (i - 1)*5, (j - 1)*5
                    if (k .eq. 1) then
                      line = 'P-AX'//string
                    else
                      line = line(1:(k - 1)*5 + 4)//string
                    end if
                    if (k .eq. 25) then
                      k = 0
                      write (punit, 320) line
320                   format (a)
                    end if
                  end if
330             continue
340           continue
              if (k .ne. 0) write (punit, 320) line
c
c  output t-axes confidence region
c
              k = 0
              do 360 i = 1, 73
                do 350 j = 1, 19
                  if (ltaxes(i, j) .eq. id) then
                    k = k + 1
                    write (string, '(i3, i2)') (i - 1)*5, (j - 1)*5
                    if (k .eq. 1) then
                      line = 'T-AX'//string
                    else
                      line = line(1:(k - 1)*5 + 4)//string
                    end if
                    if (k .eq. 25) then
                      k = 0
                      write (punit, 320) line
                    end if
                  end if
350             continue
360           continue
              if (k .ne. 0) write (punit, 320) line
              write (punit, 320) '    '
            end if
c
c accumulate statistics on fit parameter distribution for best solutions only
c
            if (best) then
              index = ifix(fit(j1, n1, m1)/.025) + 1
              if (index .gt. 20) index = 20
              nfit(index) = nfit(index) + 1
            end if
c
c recompute moment tensor representation for best solution to check for 
c polarity discrepancies.
c
            strike = phi(n1)*rad
            dip = del(j1)*rad
            slip = xlam(m1)*rad
            call shrflt (strike, dip, slip, tm)
            do 420 k = 1, nr
              if (nstat .ge. 1) then
                do 370 i = 1, nstat
                  if (stn(k) .eq. stat(i)) goto 380
370             continue
              end if
              nstat = nstat + 1
              if (nstat .gt. mxstat) then
                write (eunit, *) '***** fpfit error: # of stations ha'//
     & 've polarity discepancies exceeds ', mxstat, ' *****'
                write (eunit, *) ' station ', stn(k), ' not reported '//
     & 'in final summary'
                goto 400
              end if
              i = nstat
              stat(nstat) = stn(k)
              scnt(nstat, 1) = 0
              scnt(nstat, 2) = 0
              scntwt(nstat, 1) = 0.
              scntwt(nstat, 2) = 0.
380           read (prmk(k), '(3x, i1)') ipwt
              if (infmt .eq. 3 .and. nmsrc .gt. 0) then
	        do 381 isrc = 1, nmsrc
	          if (src(k) .eq. macsrc(isrc)) ipwt = ipwt + mxqual/2
381	        continue
	      endif
c
c recompute radiation pattern
c
              prad = 0
              do 390 l = 1, 6
                prad = prad + tm(l)*coef(k, l)
390           continue
	      if (prad .eq. 0.) prad = eps
c
c check polarity and update appropriate station count
c
              if (sign(0.5, prad) .ne. pobs(k)) then
                if (best) then
                  scnt(i, 1) = scnt(i, 1) + 1
                  scntwt(i, 1) = scntwt(i, 1) +
     & wtobs(k)*sqrt(abs(prad))
                  qcnt(ipwt + 1, 1) = qcnt(ipwt + 1, 1) + 1
                  qcntwt(ipwt + 1, 1) = qcntwt(ipwt + 1, 1) +
     & wtobs(k)*sqrt(abs(prad))
                end if
                bdflag = '*'
              else
                bdflag = ' '
              end if
              if (best) then
                scnt(i, 2) = scnt(i, 2) + 1
                scntwt(i, 2) = scntwt(i, 2) +
     & wtobs(k)*sqrt(abs(prad))
                qcnt(ipwt + 1, 2) = qcnt(ipwt + 1, 2) + 1
                qcntwt(ipwt + 1, 2) = qcntwt(ipwt + 1, 2) +
     & wtobs(k)*sqrt(abs(prad))
              end if
c
c write out to polarity file
c
400           if (lopen3) write (punit, 410) stn(k), dist(k), az(k), 
     &             ain(k), prmk(k), wtobs(k)/wtmax, bdflag, cmpnt(k),
     &             netcode(k), loc(k) 
 410          format (a5, 3f6.1, 3x, a4, f5.2, 2x, a1, 2x, a3, 1x, a2,
     &             1x, a2)
420         continue
            if (lopen3) write (punit, 320) '              '
          end if
          best = .false.
c
c end of fine search solution loop
c
430     continue
c
c end of event
c
      goto 30
      end
