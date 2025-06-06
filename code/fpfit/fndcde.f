cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C  fndcde - FiND CoDE - Returns the code for the region, ex-
C           pressed as ibit4,ibit3,ibit2,ibit1, of the screen
C           location (x,y).  See Figure 5-5, p.66, "Principles
C           of Interactive Computer Graphics", by Newman and
C           Sproull, 1979.
C
C           ibit1 = 1, if x < xleft
C           ibit2 = 1, if x > xright
C           ibit3 = 1, if y < ybot
C           ibit4 = 1, if y > ytop
C
C-
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine fndcde(ibit4,ibit3,ibit2,ibit1,x,y,xleft,xright,
     &ybot,ytop)
C
      ibit4=0
      ibit3=0
      ibit2=0
      ibit1=0
C
      If (x.LT.xleft) Then
         ibit1=1
      Else If (x.GT.xright) Then
         ibit2=1
      End If
C
      If (y.LT.ybot) Then
         ibit3=1
      Else If (y.GT.ytop) Then
         ibit4=1
      End If
C
      Return
      End
