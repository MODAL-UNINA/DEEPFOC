cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  clipper - Clips the line segment (x1,y1),(x2,y2) passed to it
C            to fit in the window defined by the x,y:
C
C               xleft<x<xright
C               ybot<y<ytop
C
C         Clipper assumes that the line segment passed to it does
C         indeed cross the given window; use tstend to determine
C         whether a line segment does indeed cross window.
C-
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine clipper(x,y,x1,y1,x2,y2,xleft,xright,ybot,ytop)
C
      delx=x2-x1
      dely=y2-y1
C
      If (abs(dely).LT.1.e-16)dely=sign(1.e-16,dely)
      If (abs(delx).LT.1.e-16)delx=sign(1.e-16,delx)
C
      slope=dely/delx
      b=y2-slope*x2
C
C - Clip along top or bottom edges.
C
      x=x1
      If (y1.GT.ytop) Then
         x=(ytop-b)/slope
         y=ytop
      Else If (y1.LT.ybot) Then
         x=(ybot-b)/slope
         y=ybot
      End If
c
c - Clip along left or right edges.
c
      If (x.GT.xright) Then
         x=xright
         y=slope*xright+b
      Else If (x.LT.xleft) Then
         x=xleft
         y=slope*xleft+b
      End If
C
      Return
      End
