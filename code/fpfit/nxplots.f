cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
C
C postscript original code from Larry Baker - modified by r saltus for hp-ux and
C character variables in symbol calls 4/92
C
C xplot.f - A (sub)set of CalComp HCBS-compatible plotting routines for
C           output to native x through the library routines by Joe Plesha
C
C converted from Larry Baker's postscript conversion by Rick Saltus 9/92
C
C
C Entry points:
C
C      Call nxPLOTS (Arguments ignored)
C
C         PLOTS  is the first plotting routine called and must be called exactly
C         once by the program
C
C         CALL PLOT (0.0,0.0,999) terminates all plotting
C
C      Call PLOT (x,y,ipen)
C
C         x     = X-coordinate
C         y     = Y-coordinate
C         ipen  = Function code
C
C                 +999 = End of run
C                   +3 = Move to (X,Y)
C                   +2 = Draw to (X,Y)
C                 -999 = End of frame
C
C                 All other values ignored
C
C      Call nxNEWPEN (ipen)
C
C         ipen  = Line color (1 to 20 pixels on an Apple LaserWriter)
C
C      Call nxSYMBOL (x, y, hgt, text, angle, nc)
C
C         x     = X-coordinate
C         y     = Y-coordinate
C         hgt   = Character height
C         text = character string or integer symbol number
C         angle = Character angle
C         nc    = Number of characters (nc>0) or integer symbol number (nc=<0)
C
C
C Notes:
C
C                                  Disclaimer
C
C Although  this program has been tested by the Geological Survey, United States
C Department of the Interior, no warranty, expressed or implied, is made by  the
C Geological  Survey,  as  to  the  accuracy  and functioning of the program and
C related program material, nor shall the fact of  distribution  constitute  any
C such  warranty,  and  no responsibility is assumed by the Geological Survey in
C connection therewith.
C
C
      Subroutine nxPLOTS (ibuf, nloc, ldev)
C
      Implicit   None
C
      common /nxbakerp/xorig,yorig
      real xorig,yorig
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,jx,jy,lastxx,lastyy
      character  junkans*1
c     Save /nxbakerp/
c     Save /xpltscale/
C
      Integer    ibuf(*), nloc, ldev
C
c     Character  title*80, create*34, for*80, afname*80
C
      real scalefactor
      common/factor0/ scalefactor

      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
C
      Data       isopen/.FALSE./
c
c     set default screen mapping in inches
c
      xinch=11.
      yinch=8.5
c
c     initialize origin offset
c
c      xorig=0.
c      yorig=0.
c     Shift origin in window a bit so that window plot looks 
c      more like postscript plot.
      xorig=0.25
      yorig=0.25
      xorig = xorig * scalefactor
      yorig = yorig * scalefactor
C
C...  Reset buffer pointer
      ipctr = 1
C
c...  findout max screen size
c
      call sizexw(jx,jy)
c     print *,'X PLOTS: Max screen size = ',jx,jy
c
c...  scale screen dimensions to match yinch/xinch ratio
c
      mpx=min(jx,800)
      mpy=int(real(mpx)*yinch/xinch)
      dpi=real(mpx)/xinch
c
c...  open x drawing window
c
c     print *,'X PLOTS: Requested window size = ',mpx,mpy
      call initxw(mpx,mpy)
c
c...  set an 8 color palette for newpen colors
c
c     print *,'X PLOTS: Setting 8 color palette'
c       call palxw(   0,   0,   0, 0 )
c       call palxw( 248, 248, 248, 1 )
        call palxw( 255,   0,   0, 0 )
        call palxw(  56, 255, 255, 1 )
c       call palxw( 255,   0,   0, 2 )
c       call palxw( 255, 128,   0, 3 )
        call palxw( 255, 128,   0, 2 )
        call palxw( 255, 255,   0, 3 )
        call palxw(   0, 128,   0, 4 )
        call palxw(   0, 128, 255, 5 )
        call palxw(   0,   0, 255, 6 )
        call palxw( 255,   0, 255, 7 )
        call palxw(   0,   0,   0, 8 )
c
c...  set drawing color = 1
c
c     print *,'X PLOTS: setting color and line style'
c     call setcolorxw(1)
c     call setlinexw(1,1,0)
C
C...  Allow file to be written by PLOT entry points now
C
      isopen = .TRUE.
C
c     print *,'X PLOTS: done with setup.'
      call plot(0.,0.,3)
      call flushxw()
      print *, ' Hit CR to continue: '
      read '(a)', junkans

      Return
C
      End
c
