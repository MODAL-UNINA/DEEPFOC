cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     GETGINPT - gets a Graphic Input Point from a terminal
C
C     This subroutine is just for native x gin (using Plesha's xdevice.c)
C
C     CALL GET_GIN_PT (itek,a,xi,yi,xhite,yhite,mode)
C
C       itek = ignored
C
C       a     = a single character returned along with location information
C
C       xi,yi = Position in plot units (inches, usually but not always)
C               of the input point on the screen
C
C       xhite,yhite = Width (xhite) and Height (yhite) of the screen in
C                     plotting units (pseudo-inches)
C
C       mode  = 0 (got new command)
C             = 1 (got valid point locations)
C
C       Called by: Getbodypt (hypermag) and other user programs
C-
      subroutine get_gin_pt(itek,a,xi,yi,xhite,yhite,mode)
      character*(*) a
C
C     Call the appropriate graphic input routine
C
      call getxgin(a,xi,yi,xhite,yhite,mode)
      return
      end
