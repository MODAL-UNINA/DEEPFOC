      subroutine fpout (ddelf, dlamf, dphif, erate, eunit, ievp, ievr,
     & ind, ires, mxdip, mxqual, mxrake, mxstat, mxstrk, ndelf, ndrng,
     & nfit, nlamf, nphif, nrev, nrrng, nsrng, nstat, qcnt, qcntwt,
     & revsta, scnt, scntwt, stat)
c
c generate summary listing of polarity discrepancies as a function of station and quality, the distribution of
c fit parameters, and the distribution of dip, strike, and rake ranges about best fit solution
c
      integer           mxqual                          
c							! (input) maximum # of qualities permitted
      integer           mxstat                          
c							! (input) maximum # of stations permitted
      real              ddelf                           
c							! (input) fault dip increment in degrees for fine search
      real              dlamf                           
c							! (input) fault rake increment in degrees for fine search
      real              dphif                           
c							! (input) fault strike increment in degrees for fine search
      real              erate(mxqual)                   
c							! (input) estimated weighted error rates
      integer           eunit                           
c							! (input) logical unit # of output of error messages
      integer           ievp                            
c							! (input) # of events processed
      integer           ievr                            
c							! (input) # of events read
      integer           ind(mxstat)                     
c							! (input) pointer array to sorted order
      integer           ires                            
c							! (input) flag: 0(1)=(un)restricted search
      integer           mxdip                           
c							! (input) maximum # of dip increments permitted
      integer           mxrake                          
c							! (input) maximum # of rake increments permitted
      integer           mxstrk                          
c							! (input) maximum # of strike increments permitted
      integer           ndelf                           
c							! (input) # of fault dip increments for fine search
      integer           ndrng(mxdip)                    
c							! (input) # of dip solution ranges binned into ddelf degree increments
      integer           nfit(20)                        
c							! (input) # of solutions binned into .025 fit increments
      integer           nlamf                           
c							! (input) # of fault rake increments for fine search
      integer           nphif                           
c							! (input) # of fault strike increments for fine search
      integer           nrev                            
c							! (input) # of reversed stations
      integer           nrrng(mxrake + 1)               
c							! (input) # of rake solution ranges binned into dlamf degree increments
      integer           nsrng(mxstrk)                   
c							! (input) # of strike solution ranges binned into dphif degree increments
      integer           nstat                           
c							! (input) total # of stations reporting for entire data set
      integer           qcnt(mxqual,2)                  
c							! (input) indx 1=# of dscrpnt plrties for qlity, indx 2=# of observations
      real              qcntwt(mxqual,2)                
c							! (input) indx 1=weighted # dscrpnt plrties for qlity, indx 2=sum of weights
      character*(*)     revsta(mxstat)                  
c							! (input) reversed station names
      integer           scnt(mxstat,2)                  
c							! (input) index 1=# of dscrpnt polarities for stat, index 2=# of obsrvations
      real              scntwt(mxstat,2)                
c							! (input) indx 1=weighted # dscrpnt polrties for stat, indx 2=sum of weights
      character*(*)     stat(mxstat)                    
c							! (input) names of all stations reporting
c
      character*2       estar                           
c							! flag indicates large discrepancy between actual and estimated error rates
      integer           i                               
c							! loop index
      integer           j                               
c							! dummy variable
      integer           k                               
c							! dummy variable
      integer           natot                           
c							! sum of # of polarities in agreement with solution
      integer           ncfit                           
c							! cumulative # of misfit scores
      integer           ndcrng                          
c							! cumulative # of solutions in dip range distribution
      integer           ndtot                           
c							! sum of # of polarities in discrepancy with solution
      integer           nrcrng                          
c							! cumulative # of solutions in rake range distribution
      integer           nscrng                          
c							! cumulative # of solutions in strike range distribution
      integer           ntot                            
c							! total # of observations
      real              rate                            
c							! weighted error rate per quality class
      character*1       star                            
c							! denotes station designated as reversed
      real              wtot                            
c							! summation of weights over all stations
c
      ndtot = 0
      ntot = 0
      wtot = 0.
      do 5 i = 1, nstat
        ndtot = ndtot + scnt(i, 1)
        ntot = ntot + scnt(i, 2)
        wtot = wtot + scntwt(i, 2)
5     continue
      natot = ntot - ndtot
c
c write out summary of polarity discrepancies by station
c
      write (eunit, 10)
10    format (/, 1x, 'SUMMARY OF STATIONS HAVING POLARITIES IN DISCREP',
     & 'ANCY WITH BEST FIT SOLUTION (* DENOTES REVERSED STATION)', /, 
     & 1x, ' STATION     DISCREPANCIES    AGREEMENTS       TOTAL ',
     & ' WEIGHTED ERROR RATE   TOTAL ERROR CONTRIBUTION', /)
c
c sort stations alphabetically
c
      call csort (stat, ind, nstat)
      do 40 i = 1, nstat
        j = ind(i)
        star = ' '
        do 20 k = 1, nrev
          if (stat(j) .eq. revsta(k)) star = '*'
20      continue
        write (eunit, 30) star, stat(j), scnt(j, 1), scnt(j, 2) -
     & scnt(j, 1), scnt(j, 2), scntwt(j, 1)/scntwt(j, 2),
     & scntwt(j, 1)/wtot
30      format (1x, a1, a5, 3(10x, i5), 9x, f6.3, 10x, f6.4)
40    continue
      write (eunit, 50) ndtot, natot, ntot
50    format (/, 1x, 'TOTAL', 3(8x, i7))
c
c write out summary of hand-picked polarity discrepancies by reading quality
c
      write (eunit, 70)
70    format (/, 1x, 'SUMMARY OF HAND-PICKED DATA WITH RESPECT TO BEST',
     & ' FIT SOLUTIONS', /, 1x, 
     & ' QUAL        DISCREPANCIES    AGREEMENTS       TOTAL     WEIG',
     & 'HTED ERROR RATE', /)
      ndtot = 0
      ntot = 0
      do 90 i = 1, mxqual/2
        estar = '  '
        ndtot = ndtot + qcnt(i, 1)
        ntot = ntot + qcnt(i, 2)
        if (qcntwt(i, 2) .eq. 0.) then
          rate = 0.
        else
          rate = qcntwt(i, 1)/qcntwt(i, 2)
          if (rate .ge. 0.0001) then
            if (abs((erate(i)-rate)/rate) .ge. 0.2) estar = '**'
          end if
        end if
        write (eunit, 80) i - 1, qcnt(i, 1), qcnt(i, 2) -
     & qcnt(i, 1), qcnt(i, 2), rate, estar
80      format (1x, 2x, i1, 2x, 3(8x, i7), 9x, f6.4, 1x, a3)
90    continue
      natot = ntot - ndtot
      write (eunit, 50) ndtot, natot, ntot
c
c write out summary of machine-picked polarity discrepancies by reading quality
c
      write (eunit, 110)
110   format (/, 1x, 'SUMMARY OF MACHINE-PICKED DATA WITH RESPECT TO B',
     & 'EST FIT SOLUTIONS', /, 1x, 
     & ' QUAL        DISCREPANCIES    AGREEMENTS       TOTAL     WEIG',
     & 'HTED ERROR RATE', /)
      ndtot = 0
      ntot = 0
      do 120 i = mxqual/2 + 1, mxqual
        estar =  '  '
        ndtot = ndtot + qcnt(i, 1)
        ntot = ntot + qcnt(i, 2)
        if (qcntwt(i, 2) .eq. 0.) then
          rate = 0.
        else
          rate = qcntwt(i, 1)/qcntwt(i, 2)
          if (rate .ge. 0.0001) then
            if (abs((erate(i)-rate)/rate) .ge. 0.2)  estar = '**'
          end if
        end if
        write (eunit, 80) i - mxqual/2 - 1, qcnt(i, 1), qcnt(i, 2) -
     & qcnt(i, 1), qcnt(i, 2), rate, estar
120   continue
      natot = ntot - ndtot
      write (eunit, 50) ndtot, natot, ntot
c
c write out distribution of fit parameters
c
      write (eunit, 130)
130   format (/, 1x, 'DISTRIBUTION OF SOLUTION MISFIT SCORES', /,
     & 1x, ' MISFIT SCORE     NUM    CUM NUM')
      ncfit = 0
      do 150 i = 1, 20
        ncfit = ncfit + nfit(i)
        write (eunit, 140) float(i - 1)*.025, float(i)*.025, nfit(i),
     1                     ncfit
140     format (1x, f5.3, ' - ', f5.3, 3x, i5, 3x, i6)
150   continue
c
c write out distribution of dip, strike, rake ranges for unrestricted searches
c
      if (ires .eq. 0) then
        write (eunit, 160)
160     format (/, 1x, 'DISTRIBUTION OF SOLUTION DIP RANGES', /,
     & 1x, ' RANGE       NUM    CUM NUM')
        ndcrng = 0
        do 180 i = 1, 10
          ndcrng = ndcrng + ndrng(i)
          write (eunit, 170) float(i - 1)*5.0, ndrng(i), ndcrng
170       format (1x, 1x, f5.1, 5x, i5, 3x, i6)
180     continue
        write (eunit, 190)
190     format (/, 1x, 'DISTRIBUTION OF SOLUTION STRIKE RANGES', /,
     & 1x, ' RANGE       NUM    CUM NUM')
        nscrng = 0
        do 200 i = 1, 19
          nscrng = nscrng + nsrng(i)
          write (eunit, 170) float(i - 1)*5.0, nsrng(i), nscrng
200     continue
        write (eunit, 210)
210     format (/, 1x, 'DISTRIBUTION OF SOLUTION RAKE RANGES', /,
     & 1x, ' RANGE       NUM    CUM NUM')
        nrcrng = 0
        do 220 i = 1, 19
          nrcrng = nrcrng + nrrng(i)
          write (eunit, 170) float(i - 1)*10.0, nrrng(i), nrcrng
220     continue
      end if
c
      write (eunit, *) 
      write (eunit, 230) ievr, ' EVENTS READ, ', ievp, ' PROCESSED'
230   format (2(i9, a))
c
      return
      end
