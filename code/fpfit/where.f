cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
      Subroutine WHERE (xnow, ynow, dfact)
C
C WHERE - Return the present location and drawing factor.
C
C Call WHERE (xnow,ynow,dfact)
C
C      xnow   = Current X-coordinate.
C      ynow   = Current Y-coordinate.
C      dfact  = Current drawing factor.
C
C    Called by:  LINE and user program
C
C        Calls:  None
C
C Commons used:  /PLTCOM/  pref,fact
C
C-
C
C     Include 'PltCom/NoList'
      Common /lbwhere/ xn,yn,df
C
C...  Return present location and drawing factor
      xnow = xn
      ynow = yn
      dfact = df
C
      Return
      End
