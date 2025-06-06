cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     ICTYPE - decides if a character is alphabetic, numeric, or other
C
C     ireturn = ICTYPE (C)
C
C            C = Single character (character*1)
C
C      ireturn =  1 , alpha (a-z,A-Z)
C              =  0 , digit (0-9)
C              = -1 , other
C
C      Called by: IGETTOK
C
C        Calls: none
C
C      Common: none
C-
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Integer Function ictype(c)
      Character*1 c,digits*10,abc*52
C
      digits='0123456789'
      abc='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
C
      If (index(abc,c).GT.0) Then
         ictype=1
         Else If (index(digits,c).GT.0) Then
         ictype=0
         Else
         ictype=-1
         End If
      Return
      End
