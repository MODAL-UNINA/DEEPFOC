C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
      SUBROUTINE LINE (XARRAY,YARRAY,NPTS,INC,LINTYP,INTEQ)
C
C LINE - connected pairs of points.
C
C CALL LINE (XARRAY,YARRAY,NPTS,INC,LINTYP,INTEQ)
C
C      XARRAY = Array of X values
C      YARRAY = Array of Y values
C      NPTS   = Number of points to be plotted
C      INC    = Increment between points in arrays
C      LINTYP = Plot control
C               < 0 = Plot markers at points
C               = 0 = Connect points by lines
C               > 0 = Plot markers at points and connect by lines
C      INTEQ  = INTEGER equivalent of marker to be used, if any
C
C Called by:  User program
C
C Calls:  PLOT, SYMBOL, WHERE
C
C COMMONs used:  None
C
C-
C
      DIMENSION  XARRAY(1), YARRAY(1)
      character*1 cinteq
C
      cinteq=char(inteq)
C
C...  Initialize subscripts
      LMIN = NPTS*INC + 1
      LDX = LMIN + INC
      NL = LMIN - INC
C
C...  Set maxs/mins and scaling factors
      FIRSTX = XARRAY(LMIN)
      DELTAX = XARRAY(LDX)
      FIRSTY = YARRAY(LMIN)
      DELTAY = YARRAY(LDX)
C
C...  Get current location
      CALL WHERE(XN,YN,DF)
      DF = AMAX1(ABS((XARRAY( 1)-FIRSTX)/DELTAX-XN),
     1           ABS((YARRAY( 1)-FIRSTY)/DELTAY-YN))
      DL = AMAX1(ABS((XARRAY(NL)-FIRSTX)/DELTAX-XN),
     1           ABS((YARRAY(NL)-FIRSTY)/DELTAY-YN))
      IPEN = 3
      ICODE = -1
      NT = IABS(LINTYP)
C
C...  No markers plotted?
      IF (LINTYP .EQ. 0) NT = 1
C
C...  Data ascending order?
      IF (DL .GE. DF) GOTO 10
C
C...  Set for descending order data
      NF = NL
      NA = ((NPTS-1)/NT)*NT + NT - (NPTS-1)
      KK = -INC
      GOTO 20
C
C...  Set for ascending order data
   10 NF = 1
      NA = NT
      KK = INC
C
C...  Markers, lines, or both?  M, L, B
   20 IF (LINTYP)              30,40,50
C
C...  Set for markers only
   30 IPENA = 3
      ICODEA = -1
      LSW = 1
      GOTO 60
C
C...  Set for lines
   40 NA = LDX
   50 IPENA = 2
      ICODEA = -2
      LSW = 0
C
C...  Plot data
   60 DO 120 I = 1,NPTS
         XN = (XARRAY(NF)-FIRSTX) / DELTAX
         YN = (YARRAY(NF)-FIRSTY) / DELTAY
C
C...  Time to plot a marker?     N, Y, N
         IF (NA - NT)           80,70,90
C
C...  Plot symbol
   70    CALL SYMBOL(XN,YN,0.08,CINTEQ,0.0,ICODE)
         NA = 1
         GOTO 110
C
C...  Line to be plotted?
   80    IF (LSW .NE. 0) GOTO 100
C
C...  Plot line
   90    CALL PLOT(XN,YN,IPEN)
C
C...  Count data point
  100    NA = NA + 1
  110    NF = NF + KK
         ICODE = ICODEA
  120    IPEN = IPENA
C
      RETURN
C
      END
