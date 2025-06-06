C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine AXIS (x,y,label,nchar,axlen,angle,fval,dv)
C
C AXIS - Plot an annotated axis.
C
C Call AXIS (x,y,label,nchar,axlen,angle,fval,dv)
C
C      (x,y)  = Starting coordinates for axis generation
C      label  = Alphanumeric text string for labelling the axis
C      nchar  = Number of characters in the axis label
C               + = Annotations are generated above the axis
C               - = Annotations are generated below the axis
C      axlen  = Axis length in inches
C      angle  = Angle in degrees at which axis is to be drawn
C      fval   = First annotation value
C      dv     = Delta annotation value
C
C    Called by:  User program
C
C        Calls:  NUMBER, PLOT, SYMBOL
C
C Commons used:  None
C
C-
C
      Parameter  (RADN = 0.01745329)
      character*5 label*(*)
C
C...  Locate which side of the axis to annotate and label
      side = +1.0
      nc = nchar
C
C...  Negative nchar?
      If (nc .lt. 0) Then
         nc = -nc
         side = -1.0
      End If
C
C..   Determine value of dv exponent
      exp = 0.0
      adv = ABS (dv)
C
C...  Zero delta annotation value?
      If (adv .ne. 0.0) Then
C...     dv exponent calculation completed?
   20    If (adv .ge. 99.0) Then
            adv = adv / 10.0
            exp = exp + 1.0
            Goto 20
         End If
C...     dv exponent calculation completed?
   30    If (adv .lt. 0.01) Then
            adv = adv * 10.0
            exp = exp - 1.0
            Goto 30
         End If
      End If
C
C...  Compute normalized fval and dv
      val = fval * (10.0**(-exp))
      adv = dv   * (10.0**(-exp))
C
C...  Set up angular orientation variables
      t2 = angle * RADN
      sina = SIN (t2)
      cosa = COS (t2)
C
      dx = -0.1
      dy = 0.15*side - 0.05
      xx = x + dx*cosa - dy*sina
      yy = y + dy*cosa + dx*sina
C
C...  Annotate axis
      ntic = axlen + 1.0
      Do 60 i = 1,ntic
         Call NUMBER (xx,yy,0.105,val,angle,2)
         val = val + adv
         xx  = xx + cosa
   60    yy  = yy + sina
C
C...  Label axis
      t2 = nc
C
C...  Does dv exponent exist?
      If (exp .ne. 0.0) t2 = nc + 6
C
      dx = -0.07*t2 + 0.5*axlen
      dy = 0.325*side - 0.075
      xx = x + dx*cosa - dy*sina
      yy = y + dy*cosa + dx*sina
      Call SYMBOL (xx,yy,0.14,label,angle,nc)
C
C...  No dv exponent to plot?
      If (exp .ne. 0.0) Then
C...     Plot exponent
         Call SYMBOL (999.,999.,0.14,'  *10',angle,5)
         t2 = nc + 5
         xx = xx + (t2*cosa - 0.8*sina)*0.14
         yy = yy + (t2*sina + 0.8*cosa)*0.14
         Call NUMBER (xx,yy,0.07,exp,angle,-1)
      End If
C
C...  Draw axis and tic marks
      dx = -0.07 * side * sina
      dy = +0.07 * side * cosa
      xx = x
      yy = y
      Call PLOT (xx,yy,3)
      Do 80 i = 1,ntic
         Call PLOT (xx,yy,2)
         Call PLOT (xx+dx,yy+dy,2)
         Call PLOT (xx,yy,3)
         xx = xx + cosa
   80    yy = yy + sina
C
      Return
C
      End
