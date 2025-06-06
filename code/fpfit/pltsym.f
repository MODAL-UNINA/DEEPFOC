      subroutine pltsym (ain, az, cx, cy, hite, name, pi, rad, rmax,
     & sym, wt)
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
c                                                       ! plot angle of symbol
      real              azr                             
c                                                       ! az in radians
      real              con                             
c                                                       ! rmax * sqrt(2.0)
      real              r                               
c                                                       ! distance from cx, cy to plot position
      real              size                            
c                                                       ! plot symbol size scaled by wt
      real              symsiz                          
c                                                       ! maximum symbol size
      real              x                               
c                                                       ! x position of symbol
      real              y                               
c                                                       ! y position of symbol
c
      parameter (ang = 0.0, symsiz = 0.2)
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
        call symbol (x, y, hite, sym, ang, 1)
c        call symbol (x, y, hite, %ref(sym), ang, 1)		! VAX/VMS version
      else
c
c first motion symbol
c
        size = symsiz*wt
        if (ain .gt. 90.) then
          call newpen (4)
        else
          call newpen (1)
        end if
        if (sym .eq. 'C') then
          call plus (size, x, y)
        else
          call circle (size, 2.0*pi, x, y)
        end if
        call newpen (1)
c
c plot station name
c
        if (name .ne. '    ') then
	  call symbol (x + size/2., y, size/2., name, 0., 4)
c	  call symbol (x + size/2., y, size/2., %ref(name), 0., 4)		! VAX/VMS version
	end if
      end if
c
      return
      end
