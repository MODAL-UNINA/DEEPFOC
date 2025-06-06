cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  plotngon - Draws an n-sided polygon centered on (x,y).  Uses the Larry 
C           Baker plot library.  NOTE: The symbol is drawn as an open shaped 
C           n-gon; and NSIDE, must be greater than or equal to three (3).
C-
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine plotngon(x,y,nside,radius,theta,nclip,xleft,xright,
     & ybot,ytop,itype)
C
cd      Print *,' Made it to plotngon'
      If (nside.GE.3) Then
         absrad = abs(radius)
         therad = theta * 3.14159 / 180.0
         thedel = 2.0 * 3.14159 / nside
         thenew = therad
         xlast = x + absrad * cos(therad)
         ylast = y + absrad * sin(therad)
C
cd         Print *,' plotting sides'
         If (nclip.EQ.1) Then
            Do 10 i = 1,nside
            thenew = thenew + thedel
            xnew = x + absrad * cos(thenew)
            ynew = y + absrad * sin(thenew)
            Call plotclip(xlast,ylast,xnew,ynew,xleft,xright,
     & ybot,ytop,itype,inout)
            xlast = xnew
            ylast = ynew
10          Continue
         Else
            Do 20 i = 1,nside
            thenew = thenew + thedel
            xnew = x + absrad * cos(thenew)
            ynew = y + absrad * sin(thenew)
            Call plotline(xlast,ylast,xnew,ynew,itype)
            xlast = xnew
            ylast = ynew
20          Continue
         End If
      End If
C
      Return
      End
