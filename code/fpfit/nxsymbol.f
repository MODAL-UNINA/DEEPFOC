cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine nxSYMBOL (x, y, hgt, text, angle, nc)
C
      Implicit   None
C
      External   PLOT
C
      common /nxbakerp/xorig,yorig
      real xorig,yorig,xt,yt
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
c
C
      Real       x, y, hgt, angle
      Character*(*) text
      Integer    nc,itext
C
      Integer    nch, xx, yy
C     Character  cbuf*255
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
c     Save /nxbakerp/
c     Save /xpltscale/
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
      If (.not. isopen) Then
         Goto 9000
      End If
C
C...
C     reset origin
C
      xt=x+xorig
      yt=y+yorig
C
C...  Scale the x and y coordinates
C
      xx = NINT (dpi * xt)
      yy = NINT (dpi * yt)
C
c     print *,'X SYMBOL: x,y,xt,yt,xx,yy = ',x,y,xt,yt,xx,yy
C
      nch=nc
      If (nch .le. 0) Then
         if (nc.lt.-1) call plot(x,y,2)
         itext=ichar(text(1:1))
c        print *,'X SYMBOL: symbol # =',itext
         Call PLSSYMB(x,y,itext,hgt)
         call flushxw ()
      Else
c        print *,'X SYMBOL: text =',text
c         cbuf=' '
c         cbuf=text(1:nch)//char(0)
c         call writexw(xx,yy,cbuf(1:nch))
         call symbl (x, y, hgt, text, angle, nc)
         call flushxw ()
      End If
c
 9000 Return
C
      End
