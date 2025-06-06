      subroutine hhog (delc, eunit, jstrt, nstrt, mstrt, igood, ipaxes,
     & itaxes, gfit, ng, idst, ndst, mxdip, mxslns, mxstrk, mxrake,
     & ndelc, nphic, nlamc, phic, xlamc, bot, irslam)
c
c  performs a "hedgehog" search through coarse solutions with fits less than fitlim, identifies solutions belonging to
c  discrete localized minima, and returns strike, dip, and rake indices of solution with best fit within each minima
c  for solutions belonging to a localized minima, compute corresponding p & t axes and set nearest grid point of "paxes", 
c  "taxes" with solution number for output of confidence region to .pol file
c
      integer           mxhog                           
c							! maximum number of solutions per localized minima
c
      parameter (mxhog = 6498)                          
c							! mxdip*mxstrk*mxrake (19*19*18)
c
      integer           mxdip                           
c							! (input) maximum number of dip values in search
      integer           mxrake                          
c							! (input) maximum number of rake values in search
      integer           mxslns                          
c							! (input) maximum number or multiple solutions permitted
      integer           mxstrk                          
c							! (input) maximum number or strike values in search
      real              bot(mxdip,mxstrk,mxrake)        
c							! (input) sum of product of observed and predicted weights
      real              delc(mxdip)                     
c							! (output) fault dip angle for coarse search
      integer           eunit                           
c							! (input) logical unit # of output of error messages
      real              gfit(mxdip*mxstrk*mxrake)       
c							! (input) contains fits of solutions in igood
      integer           idst(mxslns,5)                  
c							! (output) 1-3=indices of best fitting solutions in each localized minima
      integer           igood(mxdip*mxstrk*mxrake,4)    
c							! (input) indices of solutions with "good" fits determined by coarse search.
      integer           ipaxes(73,19)                   
c							! (output) distinct soltn # of p-axes 90% conf region 
      integer           irslam                          
c							! (input) flag: (0)1=(no) restricted coarse search range for rake angle 
      integer           itaxes(73,19)                   
c							! (output) distinct soltn # of p-axes 90% conf region 
      integer           jstrt                           
c							! (input) dip index of best solution from coarse search
      integer           mstrt                           
c							! (input) rake index of best solution from coarse search
      integer           ndelc                           
c							! (input) number of increments of dip in coarse search
      integer           ndst                            
c							! (output number of solutions in idst
      integer           ng                              
c							! (input) number of solutions in igood
      integer           nlamc                           
c							! (input) number of increments of rake in coarse search
      integer           nphic                           
c							! (input) number of increments of strike in coarse search
      integer           nstrt                           
c							! (input) strike index of best solution from coarse search
      real              phic(mxstrk)                   
c							! (input) fault strike angle in degrees for coarse search
      real              xlamc(mxrake)                   
c							! (input) fault rake angle in degrees for coarse search
c
      real              best                            
c							! largest bot for solutions with fit=fitmin (ie. ties)
      real              da1                             
c							! dip angle in degrees of priciple plane
      real              da2                             
c							! dip angle in degrees of auxilliary plane
      real              dd1                             
c							! dip directions in degrees of priciple plane
      real              dd2                             
c							! dip directions in degrees of auxilliary plane
      logical           first                           
c							! (input) flag: true=first time into subroutine 
      real              fmin                            
c							! smallest fit value within set of solutions comprising a localized minima
      integer           ic                              
c							! number of solutions used as center point for expansion
      integer           icach(mxhog)                    
c							! pointer array indices of igood
      integer           ict                             
c							! total number of solutions in a hedgehog
      integer           ig                              
c							! index over igood
      integer           ik                              
c							! loop index over icach
      integer           indxpa                          
c							! index of nearest p-axis azimuth grid element 
      integer           indxpp                          
c							! index of nearest p-axis plungle grid element 
      integer           indxta                          
c							! index of nearest t-axis azimuth grid element 
      integer           indxtp                          
c							! index of nearest t-axis plungle grid element 
      integer           j0                              
c							! dip index of center point for expansion
      integer           jj                              
c							! dip index of nearby solution to centerpoint
      integer           m0                              
c							! rake index of center point for expansion
      integer           mm                              
c							! rake index of nearby solution to centerpoint
      integer           mmx                             
c							! rake index of nearby solution to centerpoint
      integer           n0                              
c							! strike index of center point for expansion
      integer           nhh                             
c							! hedgehog index
      integer           nn                              
c							! strike index of nearby solution to centerpoint
      real              pain                            
c							! angle of incidence of p axis (deg)
      real              paz                             
c							! azimuth of p axis (deg)
      real              pi                              
c							! pi
      real              rad                             
c							! conversion from degrees to radians
      real              sa1                             
c							! slip angle in degrees of priciple plane
      real              sa2                             
c							! slip angle in degrees of auxilliary plane
      real              tain                            
c							! angle of incidence of t axis (deg)
      real              taz                             
c							! azimuth of t axis (deg)
c
      data first /.true./
      save first, pi, rad
c
      if (first) then
        first = .false.
        pi = 4.*atan(1.0)
        rad = pi/180.
      end if
c
      nhh = 1
      ic = 0
      ict = 0
      j0 = jstrt
      n0 = nstrt
      m0 = mstrt
c
c expand about (j0, n0, m0) for nearest neighbors
c
20    do 90 jj = j0 - 1, j0 + 1
        if (jj .eq. 0 .or. jj .gt. ndelc) goto 90
        do 80 nn = n0 - 1, n0 + 1
          if (nn .eq. 0 .or. nn .gt. nphic) goto 80
          do 70 mmx = m0 - 1, m0 + 1
            if ((mmx .eq.0 .or. mmx .gt. nlamc) .and. irslam .eq.1) then
              goto 70
            else if (mmx .eq. 0) then
              mm = nlamc 
            else if (mmx .gt. nlamc) then
              mm = 1
            else 
              mm = mmx
            end if
c
c look up each solution in igood. if found, annotate it with the current value of nhh
c
            do 60 ig = 1, ng
              if (igood(ig, 1) .eq. jj .and. igood(ig, 2) .eq. nn
     & .and. igood(ig, 3) .eq. mm .and. igood(ig, 4) .eq. 0) then
                igood(ig, 4) = nhh
c
c check to see if this solution is already in a cache
c
                if (ict .gt. 0) then
                  do 50 ik = 1, ict
                    if (icach(ik) .eq. ig) goto 70
50                continue
                end if
c
c store this solution in cache
c
                ict = ict + 1
                if (ict .gt. mxhog) then
                  write (eunit, *) '***** hhog error: number of solut'//
     & 'ions within hedgehog exceeds ', mxhog, ' *****'
                  stop
                end if
                icach(ict) = ig
              end if
60          continue
70        continue
80      continue
90    continue
c
c select next solution within current hedgehog as starting point for expansion
c
      ic = ic + 1
      if (ic .le. ict) then
        j0 = igood(icach(ic), 1)
        n0 = igood(icach(ic), 2)
        m0 = igood(icach(ic), 3)
        goto 20
      else
c
c finished processing cache for current hedgehog
c
        ict = 0
        ic = 0
        nhh = nhh + 1
        if (nhh .gt. mxslns) then
          print *, '***** hhog error: number of multiple solutions ex'//
     & 'ceeds ', mxslns, ' *****'
          stop
        end if
c
c get next solution from igood that does not already belong to a minima
c
        do 100 ig = 1, ng
          if (igood(ig, 4) .eq. 0) then
            j0 = igood(ig, 1)
            n0 = igood(ig, 2)
            m0 = igood(ig, 3)
            goto 20
          end if
100     continue
      end if
c
c identify solution corresponding to fit minimum within each hedgehog
c
      do 130 ndst = 1, nhh - 1
        fmin = 1.
        best = 0.
        do 120 ig = 1, ng
          if (igood(ig, 4) .eq. ndst) then
            if (gfit(ig) .lt. fmin) fmin = gfit(ig)
c
c find nearest grid cell in p, t axes arrays for 90% confidence region output 
c
            da1 = delc(igood(ig,1))
            dd1 = phic(igood(ig, 2)) + 90.
            sa1 = xlamc(igood(ig, 3))
            call auxpln (dd1, da1, sa1, dd2, da2, sa2)
            call tandp (pain, tain, paz, taz, da1, da2, dd1, dd2, sa1,
     & sa2, pi, rad)
            indxpa = nint(paz/5.) + 1
            indxpp = nint(pain/5.) + 1
            ipaxes(indxpa, indxpp) = ndst
            indxta = nint(taz/5.) + 1
            indxtp = nint(tain/5.) + 1
            itaxes(indxta, indxtp) = ndst
          end if
120     continue
c
c best solution has largest bot
c
        do 125 ig = 1, ng
          if (igood(ig, 4) .eq. ndst) then
            if (gfit(ig) .eq. fmin .and. 
     & bot(igood(ig, 1), igood(ig, 2), igood(ig, 3)) .gt. best) then
              idst(ndst, 1) = igood(ig, 1)
              idst(ndst, 2) = igood(ig, 2)
              idst(ndst, 3) = igood(ig, 3)
              best = bot(igood(ig, 1), igood(ig, 2), igood(ig, 3)) 
            end if
          end if
125     continue
130   continue
      ndst = nhh - 1
c
      return
      end
