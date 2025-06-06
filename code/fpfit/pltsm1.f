      subroutine pltsm1 (ain, az, cx, cy, hite, name, pi, rad, rmax,
     & sym, wt, pls)
c
c    plot either first motion symbol (c,d,+,-) with station name next to symbol, or stress axes symbol (p & t)
c
      real              ain                             
c                                                       ! angle of incidence of symbol
      real              az                              
c                                                       ! azimuth of symbol
      real              cx                              
c                                                       ! x position of circle center
      real              cy                              
c                                                       ! y position of circle center
      real              hite                            
c                                                       ! height of symbol
      character*4       name                            
c                                                       ! string to be plotted to right of symbol
      real              pi                              
c                                                       ! pi
      integer           pls                             
c                                                       ! 0(1)=plot +(solid circle for compressional symbol)
      real              rad                             
c                                                       ! pi/180
      real              rmax                            
c                                                       ! radius of circle
      character*1       sym                             
c                                                       ! plot symbol
      real              wt                              
c                                                       ! weight assigned to pick quality in program fpfit

      real              ainr                            
c                                                       ! ain in radians
      real              ang                             
c                                                       ! angle of circle bisector for compressional fill
      real              angle                           
c                                                       ! same as ang
      real              azr                             
c                                                       ! az in radians
      real              con                             
c                                                       ! rmax * sqrt(2.0)
      real              dang                            
c                                                       ! angle increment of circle bisector for compressional fill
      real              r                               
c                                                       ! distance from cx, cy to plot position
      real              x                               
c                                                       ! x position of symbol
      real              xpos                            
c                                                       ! x position of circle bisector for compressional fill
      real              y                               
c                                                       ! y position of symbol
      real              ypos                            
c                                                       ! y position of circle bisector for compressional fill
c
      azr = az*rad
      ainr = ain*rad
c
c upgoing rays
c
      if (ain .gt. 90.) then
        ainr = pi - ainr
        azr = pi + azr
      end if
      con = rmax*sqrt(2.0)
      r = con*sin(ainr*0.5)
      x = r*sin(azr) + cx
      y = r*cos(azr) + cy
c
c stress axis symbol
c
      if ((sym .eq. 'P') .or. (sym .eq. 'T')) then
        x = x - .286*hite
        y = y - .5*hite
        call symbol (x, y, hite, sym, 0.0, 1)
c        call symbol (x, y, hite, %ref(sym), 0.0, 1)	! VMS version
      else
c
c first motion symbol
c
        if (sym .eq. 'C') then
c
c  fill in compression circle
c
          if (pls .eq. 0) then
            call plus (hite, x, y)
          else
            call circle (hite, 2.0*pi, x, y)
            n = 9
            ang = 0.
            dang = pi/float(n)
            do 10 i = 1, n
              ang = ang + dang
              angle = ang
              xpos = x + .5*hite*cos(angle)
              ypos = y + .5*hite*sin(angle)
              call plot (xpos, ypos, 3)
              angle = ang + pi
              xpos = x + .5*hite*cos(angle)
              ypos = y + .5*hite*sin(angle)
              call plot (xpos, ypos, 2)
  10        continue
          end if
        else
          call circle (hite, 2.0*pi, x, y)
        end if
c
c plot station name
c
        if (name .ne. '    ') call symbol (x + hite/2., y, hite/2.,
     & name, 0., 4)
c        if (name .ne. '    ') call symbol (x + hite/2., y, hite/2.,	! VMS version
c     & %ref(name), 0., 4)						! VMS version
      end if
c
      return
      end
