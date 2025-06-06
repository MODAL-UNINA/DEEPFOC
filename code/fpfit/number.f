C
C+
C
      Subroutine NUMBER (x, y, height, fpn, angle, ndig)
C
C NUMBER - Plot numeric value.
C
C Call NUMBER (x,y,height,fpn,angle,ndig)
C
C      (x,y)  = Starting coordinates for 1st character.
C      height = Character height.
C      fpn    = Number to be converted to digits and plotted.
C      angle  = Angle at which numeric string is to be plotted
C               in degrees measured from the X-axis.
C      ndig   = Specification of the number of digits and the type
C               of numeric string to be plotted:
C               >  0 = Number of digits to the right of the decimal
C                      point to be plotted (last digit is rounded)
C               =  0 = Rounded integer portion of fpn is plotted
C                      with a decimal point
C               = -1 = Rounded integer portion of fpn is plotted
C                      without the decimal point
C               < -1 = Rounded integer portion of fpn is plotted
C                      after having the least significant digits
C                      truncated (IABS(ndec)-1 digits are truncated)
C
C    Called by:  User program
C
C        Calls:  TEXT
C
C COMMONs used:  None
C
C-
C
      Character  numb*10, DIGITS(0:9)
C
      Data  DIGITS/'0','1','2','3','4','5','6','7','8','9'/
C
C...  Initialize
      t1 = fpn
      xz = x
      yz = y
      nc = 1
C
C...  Number negative?
      If (t1 .lt. 0.0) Then
         t1 = -t1
         numb(1:1) = '-'
         nc = 2
      End If
C
C...  Set working digit count
      nd = -ndig
C
C...  Fractional part to be plotted?
      If (ndig .le. 0) Then
C...     Round and truncate for integer
         If (ndig .eq. 0) nd = 1
         nd = nd - 1
         t2 = IFIX((t1+0.5)/(10.0**nd)) + 0.5
         nd = 0
         If (ndig .eq. 0) nd = -1
      Else
C...     Round for fraction
         t2 = t1 + 0.5/(10.0**ndig)
      End If
C
C...  Find number of digits to the left of the decimal point
      nl = 1
C
C...  No more digits to the left of the decimal point?
   60 If (t2 .ge. 10.0) Then
         t2 = t2 / 10.0
         nl = nl + 1
         Goto 60
      End If
C
C...  Set plottable digit count
      np = nl - nd
C
C...  Bad digit count?
      If (np .le. 0) np = 1
C
C...  Time to plot decimal point?
   80 If (nl .eq. 0) Then
C...     No decimal point?
         If (ndig .lt. 0) Goto 120
         numb(nc:nc) = '.'
         If (ndig .ne. 0) np = np + 1
      Else
C...     Plot digit
         idig = t2
         t2 = (t2-idig)*10.0
         numb(nc:nc) = DIGITS(idig)
      End If
      If (nc .eq. LEN(numb)) Then
         Call TEXT (xz,yz,height,numb,angle,LEN(numb))
         xz = 999.0
         yz = 999.0
         nc = 0
      End If
C
C...  Count digit
      nc = nc + 1
      np = np - 1
  120 nl = nl - 1
C
C...  More digits to plot?
      If (np .gt. 0) Goto 80
      If (nc .gt. 1) Then
         Call TEXT (xz,yz,height,numb,angle,nc-1)
      End If
C
      Return
      End


