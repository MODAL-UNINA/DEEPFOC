      subroutine strnet (cx, cy, rad, rmax)
c
c plot perimeter of a stereo net
c
      real              cx                              
c                                                       ! x position of circle center
      real              cy                              
c                                                       ! y position of circle center
      real              rad                             
c                                                       ! pi/180
      real              rmax                            
c                                                       ! radius of circle

      real              csiz                            
c                                                       ! scratch variable (rmax/100)
      integer           i                               
c                                                       ! loop index over degrees
      integer           n                               
c                                                       ! tests 10 degree tick position
      integer           nn                              
c                                                       ! tests 90 degree tick position
      real              p                               
c                                                       ! tick length
      real              phi                             
c                                                       ! azimuth in radians
      real              x                               
c                                                       ! x postion of circle
      real              xp                              
c                                                       ! x position of end of tick
      real              y                               
c                                                       ! y postion of circle
      real              yp                              
c                                                       ! y position of end of tick
c
      call newpen (2)
c
c draw circle @ 1 degree increments
c
      do 10 i = 1, 361
        phi = (i - 1)*rad
        x = rmax*cos(phi) + cx
        y = rmax*sin(phi) + cy
        n = (i - 1) - ((i - 1)/10)*10
        nn = (i - 1) - ((i - 1)/90)*90
        p = 0.01*rmax
        if ((n .eq. 0) .and. (i .gt. 10)) p = 0.02*rmax
        if ((nn .eq. 0) .and. (i .gt. 90)) p = 0.04*rmax
        xp = (rmax + p)*cos(phi) + cx
        yp = (rmax + p)*sin(phi) + cy
        if (i .gt. 1) then
          call plot (x, y, 2)
          call plot (xp, yp, 2)
          call plot (x, y, 2)
        else
          call plot (x, y, 3)
        end if
10    continue
c
c plot + at center
c
      csiz = .01*rmax
      call plot (cx - csiz, cy, 3)
      call plot (cx + csiz, cy, 2)
      call plot (cx, cy - csiz, 3)
      call plot (cx, cy + csiz, 2)
      call newpen (1)
c
      return
      end
