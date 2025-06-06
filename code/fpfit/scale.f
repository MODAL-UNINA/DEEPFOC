C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
      SUBROUTINE SCALE (ARRAY,AXLEN,NPTS,INC)
C
C SCALE - Scale data values for LINE and AXIS.
C
C CALL SCALE (ARRAY,AXLEN,NPTS,INC)
C
C      ARRAY  = Array of unscaled data points
C      AXLEN  = Length of scaled axis in inches
C      NPTS   = Number of data points to be scaled
C      INC    = Increment between points in ARRAY
C
C Called by:  User program
C
C Calls:  None
C
C COMMONs used:  None
C
C-
C
      DIMENSION  ARRAY(1), UNITS(7)
C
      DATA UNITS(1)/1./,UNITS(2)/2./,UNITS(3)/4./,UNITS(4)/5./
      DATA UNITS(5)/8./,UNITS(6)/10./,UNITS(7)/20./
C
C...  Determine min and max values of ARRAY
      K = IABS(INC)
      J = NPTS * K
      AMIN = ARRAY(1)
      AMAX = AMIN
      DO 10 I = 1,J,K
         A = ARRAY(I)
C
C...  New min found?
         IF (A .LT. AMIN) AMIN = A
C
C...  New max found?
         IF (A .GT. AMAX) AMAX = A
   10    CONTINUE
C
C...  Compute delta value for unscaled unit interval
      DV = (AMAX-AMIN) / AXLEN
C
C...  Delta OK?
      IF (DV .GT. 0.0) GOTO 20
C
C...  Error - AMAX and AMIN not in correct range
      DV = ABS((AMIN+AMIN)/AXLEN) + 1.0
C
C...  Compute tens power of DV value (watch out for negative logs!)
   20 ADV = ALOG10(DV)
      LOGDV = IFIX(ADV)
      IF (FLOAT(LOGDV) .GT. ADV) LOGDV = LOGDV - 1
      A = 10.0**LOGDV
C
C...  Compute normalized DV value (1. < DV < 10.)
      DV = DV/A
C
C...  Locate closest "UNIT" DV value (normalized)
      DO 30 I = 1,6
C
C...  Value found?
         IF (UNITS(I) .GE. DV) GOTO 50
   30    CONTINUE
C
C...  Expand unit DV to floating
   50 DV = UNITS(I) * A
C
C...  Compute "UNIT" minimum based on "UNIT" DV
      ADV = AINT(AMIN/DV)
      IF (ADV .GT. AMIN/DV) ADV = ADV - 1.
      TMIN = DV * ADV
C
C...  Does adjusted "UNIT" scale fit axis length?
      IF ((TMIN+(AXLEN+0.01)*DV) .GE. AMAX) GOTO 60
      ADV = AINT(AMIN/A)
      IF (ADV .GT. AMIN/A) ADV = ADV - 1.
      TMIN = A * ADV
C
C...  Does adjusted "UNIT" scale fit axis length?
      IF ((TMIN+(AXLEN+0.01)*DV) .GE. AMAX) GOTO 60
      I = I + 1
      GOTO 50
C
C...  Compute final adjusted minimum
   60 TMIN = TMIN - DV*AINT((AXLEN+(TMIN-AMAX)/DV)/2.0)
C
C...  Does TMIN need correction?
      IF (AMIN*TMIN .LE. 0.0) TMIN = 0.0
C
C...  Scale direction OK?
      IF (INC .GT. 0) GOTO 70
C
C...  Reverse direction of scale
      TMIN = TMIN + DV*AINT(AXLEN+0.5)
      DV = -DV
C
C...  Set scale factors into user's array
   70 J = J + 1
      ARRAY(J) = TMIN
      K = J + K
      ARRAY(K) = DV
C
      RETURN
C
      END
