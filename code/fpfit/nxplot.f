cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine nxPLOT (x, y, ipen)
C
      Implicit   None
C
      character*1 junkans

      common /nxbakerp/xorig,yorig
      real xorig,yorig,xt,yt
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
c     Save /nxbakerp/
c     Save /xpltscale/
C
      Real       x, y
      Integer    ipen
C
      Integer    xx, yy
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
C
      If (.not. isopen) Then
         Goto 9000
      End If
C
C...  offset origin
C
      xt=x+xorig
      yt=y+yorig
C
C...     reset origin if ipen<0
C
      if (ipen.lt.0) then
c
c     reset origin
c
c        print *,'PLOT - origin reset'
         xorig=xorig + x
         yorig=yorig + y
      end if
C
C...  Scale the x and y coordinates
C
      xx = nint (dpi * xt)
      yy = nint (dpi * yt)
C
      if ((xx.gt.mpx).or.(yy.gt.mpy)) then
        print *,'X PLOT: outside window x,y = ',xx,yy
      end if
C
      If (ipen .eq. 2) Then
C
C...     DRAW command
c
c        draw line in x window
c
c        print *,'X PLOT: linexw ',lastxx,lastyy,xx,yy
         call linexw(lastxx,lastyy,xx,yy,1,2)
         lastxx=xx
         lastyy=yy
c
C...     Mark this frame dirty
c        ismove = .FALSE.
c        dirty  = .TRUE.
C
      Else If ((ipen .eq. 3) .or. (ABS(ipen) .eq. 999)) Then
C
C...     MOVE or END PLOT command
C
c        ismove = .TRUE.
C
C...     Print the page if this is the END OF FRAME/RUN
C
         If (ABS(ipen) .eq. 999) Then
               call flushxw
               lastxx = 0
               lastyy = 0
c              virgin = .TRUE.
               if (ipen .eq. -999) then
                 print *, ' Hit CR to continue: '
                 read '(a)', junkans
                 call clearxw ()
               endif
C
         End If
C
C...     Flush buffers and close file if this is the END OF RUN
C...     (or the END OF FRAME in the Encapsulated PostScript version)
C
         If (ipen .eq. 999) Then
c           Call TRAILR
            call closexw
            Goto 9000
         End If
C
      End If
C
C...  Save last coordinate positions for initiation of a sequence of DRAW
C...  commands or the continuation of DRAW commands that span multiple paths.
      lastxx = xx
      lastyy = yy
C
 9000 Return
C
      End
