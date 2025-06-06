C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
      SUBROUTINE MPLINE2 (N,X,Y,XMIN,XMAX,XL,YMIN,YMAX,YL,
     1                   LINTYP,MARKER,SIZE,IPEN,SVAREA)
C
C MPLINE2 -- Poly-line drawing subroutine
C
C           Draw a line through a set of points, with an optional
C           marker displayed at each point.  (The line may be
C           invisible.)  Clipping is performed using user supplied
C           limits.
C
C CALL MPLINE2(N,X,Y,XMIN,XMAX,XL,YMIN,YMAX,YL,
C             LINTYP,MARKER,SIZE,IPEN,SVAREA)
C
C      N      = Number of data points.
C      X      = X coordinates of data points.
C      Y      = Y coordiantes of data points.
C      XMIN,XMAX = X clipping limits, in data units.
C      XL     = Length of X axis, in plotting units.
C      YMIN,YMAX = Y clipping limits, in data units.
C      YL     = Length of Y axis, in plotting units.
C      LINTYP = Line type flag:  <0 -> Plot markers at points
C                                =0 -> Connect points by lines
C                                >0 -> Connect markers at points with lines
C      MARKER = Symbol number
C      SIZE   = Marker size
C      IPEN   = Move to first point with pen up (IPEN=3) or down (IPEN=2)
C      SVAREA = Real array length 2 to save previous position (in plotting
C               units) for next call to MPLINE2 .
C               SVAREA must be valid if IPEN=2.
C
C Note:  The data value (XMIN,YMIN) is mapped to plotting position (0.,0.);
C        data value (XMAX,YMAX) is mapped to plotting position (XL,YL).
C
C      Called by: User program
C
C          Calls: Plot, Symbol, Plclip
C
C   Commons used: none
C-
C
      DIMENSION  X(1), Y(1), SVAREA(2), VECTOR(4), WINDOW(4)
      character*1 cmarker
C
      PX(XV) = (XV-XMIN) / DX
      PY(YV) = (YV-YMIN) / DY
C
      cmarker=char(marker)
C
      DX = (XMAX-XMIN) / XL
      DY = (YMAX-YMIN) / YL
      WINDOW(1) = 0.0
      WINDOW(2) = 0.0
      WINDOW(3) = XL
      WINDOW(4) = YL
C...  Make sure N >= 1
      IF (N .LE. 0) GOTO 9000
C...  Get previous position from save area
      XP = PX(XMIN)
      YP = PY(YMIN)
      JPEN = 3
      IF (LINTYP .LT. 0) GOTO 1000
      IF (IPEN .NE. 2) GOTO 1000
      XP = SVAREA(1)
      YP = SVAREA(2)
      JPEN = 2
C...  Do loop 1,N
 1000 DO 3000 J = 1,N
         VECTOR(1) = XP
         VECTOR(2) = YP
         VECTOR(3) = PX(X(J))
         VECTOR(4) = PY(Y(J))
         XP = VECTOR(3)
         YP = VECTOR(4)
C...  Clip vector
         CALL PLCLIP(VECTOR,WINDOW,NCLIP)
         IF (NCLIP .GE. 3) GOTO 3000
C...  End points for calls to PLOT and SYMBOL
         XT = VECTOR(3)
         YT = VECTOR(4)
C...  If line, plot line
         IF (LINTYP .LT. 0) GOTO 2000
         CALL PLOT(VECTOR(1),VECTOR(2),3)
         CALL PLOT(XT,YT,JPEN)
         IF (LINTYP .EQ. 0) GOTO 3000
C...  If end point not clipped, plot symbol
 2000    IF (VECTOR(3) .NE. XP) GOTO 3000
         IF (VECTOR(4) .EQ. YP)
     1      CALL SYMBOL(XT,YT,SIZE,CMARKER,0.0,-1)
 3000    JPEN = 2
C
C...  Save last point position for next call to MPLINE2.
      SVAREA(1) = XP
      SVAREA(2) = YP
C
C...  Return
 9000 RETURN
C
      END
