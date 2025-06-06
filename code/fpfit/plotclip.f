cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  plotclip - Clips and draws a line segment passed to the routine using the 
C             Larry Baker plot library passed to the routine.
C
C-
      Subroutine plotclip(x1,y1,x2,y2,xleft,xright,ybot,ytop,itype,
     & inout)
C
      Call tstend(inout,x1,y1,x2,y2,xleft,xright,ybot,ytop)
C
      If (inout.EQ.3) Then
         Call plotline(x1,y1,x2,y2,itype)
      Else If (inout.EQ.2) Then
         Call clipper(xone,yone,x1,y1,x2,y2,xleft,xright,ybot,ytop)
         Call plotline(xone,yone,x2,y2,itype)
      Else If (inout.EQ.1) Then
         Call clipper(xtwo,ytwo,x2,y2,x1,y1,xleft,xright,ybot,ytop)
         Call plotline(x1,y1,xtwo,ytwo,itype)
      Else If (inout.EQ.0) Then
         Call clipper(xone,yone,x1,y1,x2,y2,xleft,xright,ybot,ytop)
         Call clipper(xtwo,ytwo,x2,y2,x1,y1,xleft,xright,ybot,ytop)
         Call plotline(xone,yone,xtwo,ytwo,itype)
      End If
C
      Return
      End
