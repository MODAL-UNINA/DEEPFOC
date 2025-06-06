      subroutine plus (size, x0, y0)
c
c plot a "plus" sign centered at x0, y0
c
      real              size                            
c                                                       ! size of triange
      real              x0                              
c                                                       ! x postion of center
      real              y0                              
c                                                       ! y postion of center
c
      real              x                               
c                                                       ! x plot postion
      real              y                               
c                                                       ! y plot postion
c
c move to top
c
      y = y0 + size/2.
      call plot (x0, y, 3)
c
c draw to bottom
c
      y = y - size
      call plot (x0, y, 2)
c
c move to right
c
      x = x0 + size/2.
      call plot (x, y0, 3)
c
c draw to left
c
      x = x - size
      call plot (x, y0, 2)
c
      return
      end
