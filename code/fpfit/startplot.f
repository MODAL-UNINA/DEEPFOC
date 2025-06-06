cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
C nplot.f - A (sub)set of CalComp HCBS-compatible plotting routines for
C           output to selected device routines 
C
C ---------------------------------------------------------------------------
C r saltus 12/24/92 - conversion from xplot.f to switch between x-window and
C                     postscript drivers
C 
C ---------------------------------------------------------------------------
C
C Entry points:
C
C      Call PLOTS (Arguments ignored)
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
C      Call NEWPEN (ipen)
C
C         ipen  = Line color (1 to 20 pixels on an Apple LaserWriter)
C
C      Call SYMBOL (x, y, hgt, text, angle, nc)
C
C         x     = X-coordinate
C         y     = Y-coordinate
C         hgt   = Character height
C         text = character string or integer symbol number
C         angle = Character angle
C         nc    = Number of characters (nc>0) or integer symbol number (nc=<0)
C
C
c Notes:
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
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine STARTPLOT (iplotr)
      Implicit   None
      integer iplotr
      integer nxplotr, npplotr
      common /npswitch/ nxplotr, npplotr

      if (iplotr .eq. 1) then
          nxplotr = 1
          npplotr = 0
      else if (iplotr .eq. 2) then
          nxplotr = 0
          npplotr = 1
      else if (iplotr .eq. 3) then
          nxplotr = 1
          npplotr = 1
      else if (iplotr .eq. 0) then
          nxplotr = 0
          npplotr = 0
      else
          print *, ' ** iplotr not understood in STARTPLOT'
      endif

      if ((nxplotr.ne.0).or.(npplotr.ne.0)) then
        call plots(0,0,0)
      endif

      return
      end
