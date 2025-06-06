cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  tstend - TeST END - Tests the endpoints of the line (x1,y1),
C           (x2,y2) for mapping onto region 0000 (See "Principles
C           of Interactive Computer Graphics", by Newman and
C           Sproull, figure 5-5, page 66.)
C
C           Region 0000 is defined as any x,y:
C              xleft < x < xright
C              ybot  < y < ytop
C
C             inout = Value describing properties of pair
C                 3 = Both points in region 0000
C                 2 = point (x2,y2) within region 0000, (x1,y1) outside
C                 1 = point (x1,y1) within region 0000, (x2,y2) outside
C                 0 = line segment crosses region 0000 (endpoints
C                     are outside)
C                -1 = line segment is entirely off screen
C
C-
      Subroutine tstend(inout,x1,y1,x2,y2,xleft,xright,ybot,ytop)
C
      If (x1.GT.x2) Then
         xmn = x2
         xmx = x1
      Else
         xmn = x1
         xmx = x2
      End If
C
      If (y1.GT.y2) Then
         ymn = y2
         ymx = y1
      Else
         ymn = y1
         ymx = y2
      End If
C
      If (xleft.GT.xright) Then
         xl = xright
         xr = xleft
      Else
         xr = xright
         xl = xleft
      End If
C
      If (ybot.GT.ytop) Then
         yb = ytop
         yt = ybot
      Else
         yt = ytop
         yb = ybot
      End If
C
C - Take care of the easy cases when both points are inside or both ponts
C   outside of the region.
C
      If (xmx.LE.xr.AND.xmn.GE.xl.AND.ymx.LE.yt.AND.ymn.GE.yb) Then
         inout = 3
         Return
      Else
         If ((xmx.LT.xl.OR.xmn.GT.xr).OR.
     &       (ymx.LT.yb.OR.ymn.GT.yt)) Then
            inout = -1
            Return
         End If
      End If
C
      Call fndcde(ione4,ione3,ione2,ione1,x1,y1,xl,xr,yb,yt)
      Call fndcde(itwo4,itwo3,itwo2,itwo1,x2,y2,xl,xr,yb,yt)
C
      nsum1 = ione4 + ione3 + ione2 + ione1
      nsum2 = itwo4 + itwo3 + itwo2 + itwo1
C
      If (nsum1.EQ.0.AND.nsum2.EQ.0) Then
         inout = 3
      Else If (nsum1.EQ.0) Then
         inout = 1
      Else If (nsum2.EQ.0) Then
         inout = 2
      Else
         inout = -1
         If ((ione4.EQ.1.AND.itwo4.EQ.1).OR.
     &       (ione3.EQ.1.AND.itwo3.EQ.1).OR.
     &       (ione2.EQ.1.AND.itwo2.EQ.1).OR.
     &       (ione1.EQ.1.AND.itwo1.EQ.1)) Return
C
         delx = x2 - x1
         dely = y2 - y1
         If (abs(delx).LT.1.e-18) delx = sign(1.e-18,delx)
         If (abs(dely).LT.1.e-18) dely = sign(1.e-18,dely)
C
         slope = dely / delx
         b = y2 - slope * x2
C
         ytest = slope * xl + b
         If (ytest.LT.yb.OR.ytest.GT.yt) Go To 10
         inout = 0
         Return
   10    Continue
C
         ytest = slope * xr + b
         If (ytest.LT.yb.OR.ytest.GT.yt) Go To 20
         inout = 0
         Return
   20    Continue
C
         xtest = (yb - b) / slope
         If (xtest.LT.xl.OR.xtest.GT.xr) Go To 30
         inout = 0
         Return
   30    Continue
C
         xtest = (yt - b) / slope
         If (xtest.LT.xl.OR.xtest.GT.xr) Go To 40
         inout = 0
   40    Continue
      End If
C
      Return
      End
