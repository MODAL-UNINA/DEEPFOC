      subroutine search (bot, coef, ddel, del, delc, del0, dfit, dlam,
     & dphi, eps, first, fit, fitmin, flag, gfit, iamwt, igood, j1,
     & m1, mxdip, mxrake, mxstat, mxstrk, n1, ndel, ng, nlam, nphi, nr,
     & phis, phisc, phis0, pobs, rad, wtobs, xlam, xlamc, xlam0)
c
c loop over the entire focal mechanism space, compute fit parameter for each solution, and return best fit indices
c in case of tie fit, choose fit with largest "bot".  if fine search (first = false) then fill in flag array with
c stars for solutions with fit parameter <= best fit + dfit
c
      integer           mxdip                           
c							! (input) maximum # of dip increments permitted
      integer           mxrake                          
c							! (input) maximum # of rake increments permitted
      integer           mxstat                          
c							! (input) maximum # of stations permitted
      integer           mxstrk                          
c							! (input) maximum # of strike increments permitted
      real              bot(mxdip,mxstrk,mxrake)        
c							! (output) sum of product of observed and predicted weights
      real              coef(mxstat,6)                  
c							! (input) coefficients by which tm multiplied to give p radiation pattern
      real              ddel                            
c							! (input) fault dip increment in degrees
      real              del(mxdip)                      
c							! (output) fault dip angle in degrees
      real              delc(mxdip)                     
c							! (output) fault dip angle for coarse search
      real              del0                            
c							! (input) initial fault dip angle in degrees
      real              dfit                            
c							! (input) increment to fit function
      real              dlam                            
c							! (input) fault rake increment in degrees
      real              dphi                            
c							! (input) fault strike increment in degrees
      real              eps
c                                                       ! (input) machine precision
      logical           first                           
c							! (input) flag: true=first time into subroutine search
      real              fit(mxdip,mxstrk,mxrake)        
c							! (output) weighted measure of agreement between obs, pred polarities
      real              fitmin                          
c							! (output) fit of best solution corresponding to fit(j1, n1, m1)
      character*1       flag(mxdip,mxstrk,mxrake)       
c							! (output) if fit < fitlim then '*', otherwise blank
      real              gfit(mxdip*mxstrk*mxrake)       
c							! (output) fits of good solutions from coarse search
      integer           iamwt                           
c							! (input) flag controling amplitude weighting (0=omit, 1=use)
      integer           igood(mxdip*mxstrk*mxrake,4)    
c							! (output) array containing indices of good solutions (coarse)
      integer           j1                              
c							! (output) dip index of best solution
      integer           m1                              
c							! (output) rake index of best solution
      integer           n1                              
c							! (output) strike index of best solution
      integer           ndel                            
c							! (input) number of fault dip increments
      integer           ng                              
c							! (output) number of good solutions in coarse search
      integer           nlam                            
c							! (input) number of fault rake increments
      integer           nphi                            
c							! (input) number of fault strike increments
      integer           nr                              
c							! (input) -1=eof, 0=nr<minobs, nr>0 => number of stations
      real              phis(mxstrk)                    
c							! (output) fault strike angle in degrees
      real              phisc(mxstrk)                   
c							! (output) fault strike angle in degrees for coarse search
      real              phis0                           
c							! (input) initial fault strike angle in degrees
      real              pobs(mxstat)                    
c							! (input) observed first motion polarities; .5=compression, -.5=dilatation
      real              rad                             
c							! (input) conversion from degrees to radians
      real              wtobs(mxstat)                   
c							! (input) observed first motions weights
      real              xlam(mxrake)                    
c							! (output) fault rake angle in degrees
      real              xlamc(mxrake)                   
c							! (output) fault rake angle in degrees for coarse search
      real              xlam0                           
c							! (input) initial fault rake angle in degrees
c
      real              best                            
c							! largest bot for solutions with fit=fitmin (ie. ties)
      real              bot1                            
c							! sum of product of observed weights
      real              dip                             
c							! fault dip angle in radians
      real              fitlim                          
c							! upper bound on "good" solutions in search
      integer           i                               
c							! loop index 
      integer           j                               
c							! loop index over dip
      integer           k                               
c							! loop index 
      integer           m                               
c							! loop index over rake
      integer           n                               
c							! loop index of strike
      real              prad                            
c							! radiation amplitude corresponding ain, phi.
                                                        
c							! (dilatation) -1.<prad<+1.(compression)
      real              pth                             
c							! predicted first motion polarity; same convention as for pobs
      real              slip                            
c							! fault slip angle in radians
      real              strike                          
c							! fault strike angle in radians
      real              tm(6)                           
c							! moment tensor in upper triangular symetric storage mode
      real              top                             
c							! sum of amp wtd difference of predicted, obs. polarities; 0<= top <=1
      real              top1                            
c							! sum of non-amp wtd difference of predicted, obs. polarities; 0<= top <=1
      real              wtth                            
c							! predicted first motions weights
c
      best = 0.
      fitmin = 2.0
      do 50 m = 1, nlam
        xlam(m) = xlam0 + (m - 1)*dlam
        if (first) xlamc(m)=xlam(m)
        do 40 n = 1, nphi
          phis(n) = phis0 + (n - 1) * dphi
          if (first) phisc(n)=phis(n)
          do 30 j = 1, ndel
            del(j) = del0 + (j - 1) * ddel
            if (first) delc(j)=del(j)
            strike = phis(n)*rad
            dip = del(j)*rad
            slip = xlam(m)*rad
c
c calculate moment tensor representation of shear fault (postive up, south, east)
c
            call shrflt (strike, dip, slip, tm)
c
c calculate radiation pattern for model (eqtn 4.91, aki & richards, pg. 118)
c
            top = 0
            bot(j, n, m) = 0
            top1 = 0.
            bot1 = 0.
            do 20 i = 1, nr
              prad = 0
              do 10 k = 1, 6
                prad = prad + tm(k)*coef(i, k)
10            continue
c
c select amplitude weighting and calculate fit function for this model
c
              pth = sign(0.5, prad)
              wtth = sqrt(abs(prad))
              top = top + abs(pobs(i) - pth)*wtobs(i)*wtth
              bot(j, n, m) = bot(j, n, m) + wtobs(i)*wtth
              top1 = top1 + abs(pobs(i) - pth)*wtobs(i)
              bot1 = bot1 + wtobs(i)
20          continue
c
c use amplitude weighting
            if (iamwt .eq. 1) then
              fit(j, n, m) = top/bot(j, n, m)
c
c do not use amplitude weighting
            else if (iamwt .eq. 0) then
              fit(j, n, m) = top1/bot1
            end if
c
            if (fit(j, n, m) .lt. fitmin) fitmin = fit(j, n, m)
30        continue
40      continue
50    continue
c
c for tie solutions, find solution with most stations away from nodes
c
      fitlim = fitmin + dfit
      if (first) ng = 0
      do 90 m = 1, nlam
        do 80 n = 1, nphi
          do 70 j = 1, ndel
            if (((fit(j, n, m) - fitmin) .lt. eps) .and. bot(j, n, m)
     & .gt. best) then
              best = bot(j, n, m)
              j1 = j
              n1 = n
              m1 = m
            end if
c
c star solutions having fit within dfit of fitmin 
c
            if (fit(j, n, m) .le. fitlim) then
              flag(j, n, m) = '*'
c
c save solutions in coarse search with fit .le. fitlim as "good" solutions
c
              if (first) then
                ng = ng + 1
                igood(ng, 1) = j
                igood(ng, 2) = n
                igood(ng, 3) = m
                igood(ng, 4) = 0
                gfit(ng) = fit(j, n, m)
              end if
            else
              flag(j, n, m) = ' '
            end if
70        continue
80      continue
90    continue
      flag(j1, n1, m1) = 'a'
c
      return
      end
