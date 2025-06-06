cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  plotline - Plots a line segment using the Larry Baker plot library.
C             NOTE: It is recommended to use a Call to PLOTCLIP rather than
C             direct calls to PLOTLINE.  This will prevent lines from being
C             drawn off of a "window".
C
C-
      Subroutine plotline(x1,y1,x2,y2,incr)
C
      If (incr.EQ.0) Then
         Call plot(x1,y1,+3)
         Call plot(x2,y2,+2)
      Else
         iflag = 1
         If (incr.LT.0) iflag = 0
         del = abs(incr / 100.0)
         delx = x2 - x1
         dely = y2 - y1
         dist = sqrt(delx**2 + dely**2)
         ncount = int(dist / del)
         If (ncount.LE.0.AND.iflag.EQ.1) Then
            Call plot(x1,y1,+3)
            Call plot(x2,y2,+2)
         Else If (ncount.GE.1) Then
            test = dist - ncount * del
            idraw = 1
            If (test.GT.0.0.AND.test.LT.0.01) idraw = 0
            xadd = delx / ncount
            yadd = dely / ncount
            xp1 = x1
            yp1 = y1
            Do 10 i = 1,ncount
            xp2 = xp1 + xadd
            yp2 = yp1 + yadd
            iodd = mod((i+iflag),2)
            If (iodd.EQ.0) Then
               Call plot(xp1,yp1,+3)
               Call plot(xp2,yp2,+2)
            End If
            xp1 = xp2
            yp1 = yp2
10          Continue
            If (iodd.EQ.1.AND.idraw.EQ.1) Then
               Call plot(xp2,yp2,+3)
               Call plot(x2,y2,+2)
            End If
         End If
      End If
C
      Return
      End
