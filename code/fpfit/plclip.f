C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C
      Subroutine PLCLIP (vector, window, iclip)
C
C PLCLIP - Clip vector to plotting window.
C
C Call PLCLIP (vector,window,iclip)
C
C      vector = The vector to be clipped (in place)
C               (vector(1,1),vector(2,1)) to (vector(1,2),vector(2,2))
C      window = Clipping limits (X-Min,Y-Min), (X-Max,Y-Max)
C      iclip  = 0 If vector good
C             = 1 If vector clipped once
C             = 2 If vector clipped twice
C             = 3 If vector bad
C
C       Called by:  PLPLOT, PLINE1, PLINE2
C
C           Calls:  None
C
C Parameters used:  None
C
C    Commons used:  None
C
C-
C
      Dimension  vector(2,2), window(2,2)
C
C...  Vector limits check and clipping
C
C...  Test each coordinate
      iclip = 0
      Do 100 i = 1,2
         nerr = 0
C
C...     Test individual ordinates
   50    Do 70 j = 1,2
C
C...        Test upper limits
            almt = window(j,2)
C
C...        Upper limit OK?
            If (vector(j,i) .gt. almt) Then
C
C...           Upper limit bad?
               If (vector(j,3-i) .gt. almt) Then
                  Goto 9900
               Else
                  Goto 90
               End If
            End If
C
C...        Test lower limits
            almt = window(j,1)
C
C...        Lower limit OK?
            If (vector(j,i) .lt. almt) Then
C
C...           Lower limit bad?
               If (vector(j,3-i) .lt. almt) Then
                  Goto 9900
               Else
                  Goto 90
               End If
            End If
   70       Continue
         Goto 100
C
C...     Ordinate out of range - attempt intersection
   90    nerr = nerr + 1
C
C...     Intersection failure?
         If (nerr .gt. 2) Then
            Goto 9900
         Else
            vector(3-j,i) = ((vector(3-j,1)-vector(3-j,2))/
     1                       (vector(  j,1)-vector(  j,2)))*
     2                       (almt-vector(j,1)) + vector(3-j,1)
            vector(  j,i) = almt
            Goto 50
         End If
C
C...     Test for any clipping done for this point
  100    If (nerr .gt. 0) iclip = iclip + 1
C
 9000 Return
C
C...  Intersection failure
 9900 iclip = 3
      Goto 9000
C
      End
