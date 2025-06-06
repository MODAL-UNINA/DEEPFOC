      subroutine circle (size, twopi, x0, y0)
c
c plot a circle
c
      real              size                            
c                                                       ! size of circle
      real              twopi                           
c                                                       ! two*pi
      real              x0                              
c                                                       ! x postion of center
      real              y0                              
c                                                       ! y postion of center
c
      real              angle                           
c                                                       ! angle
      integer           j                               
c                                                       ! loop index
      integer           n                               
c                                                       ! number of points of which circle composed
      real              size2                           
c                                                       ! scratch variable
      real              x                               
c                                                       ! x plot postion
      real              y                               
c                                                       ! y plot postion
c
      size2 = size*0.5
c
c compute optimum # of points to draw
c
      n = 20*sqrt(size2*20.)
      if (n .lt. 10) n = 10
c
c draw circle
c
      x = x0 + size2
      call plot (x, y0, 3)
      do 10 j = 1, n
        angle = twopi*float(j)/float(n)
        x = x0 + size2*cos(angle)
        y = y0 + size2*sin(angle)
        call plot (x, y, 2)
10    continue
c
      return
      end
