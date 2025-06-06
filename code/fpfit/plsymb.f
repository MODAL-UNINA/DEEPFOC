C
C+
C
      Subroutine PLSYMB (xz, yz, hgt, itext, angle, nc, PLOT)
C
C Custom version of PLSYMB for drawing centered markers only (intended for
C use with PostScript.for to provide a complete, non-interactive, CalComp-
C compatible subroutine library that produces PostScript output files.)
C
C
C Call PLSYMB (x,y,hgt,itext,angle,nc,PLOT)
C
C      (x,y)  = Center coordinate of marker.
C      hgt    = Character height specification (inches).
C               If hgt <= 0.0, previous hgt and angle are used.
C      itext  = Marker no. (0-13).
C      angle  = Angle at which the marker is plotted.
C      nc     = Number of characters to be plotted.
C               = -1 =  Move to (x,y) with pen up; plot marker.
C               < -1 =  Move to (x,y) with pen down; plot marker.
C      PLOT   = External CalComp-compatible PLOT subroutine.
C
C
C    Called by:  SYMBOL
C
C        Calls:  PLOT (passed as an argument)
C
C Commons used:  None
C
C-
C
      External   PLOT
C
      Parameter  ( RADCO = 0.01745329 )
      Parameter  ( EPSIL = 0.0000277  )
C
      Dimension  xa(0:4), ya(0:4), asin(5), acos(5)
      Dimension  khar(0:14), nodes(2,0:119)
C
      Save  theta, facc, xa, ya
C
      Data  facc/0.0/, theta/0.0/
      Data  ancc/1.0/, ancs/0.0/, xt/0.0/, yt/0.0/
      Data  xa/5*0.0/, ya/5*0.0/
      Data  asin/0.0,1.0,0.0,-1.0,0.0/
      Data  acos/1.0,0.0,-1.0,0.0,1.0/
      Data  khar /0,8,20,26,34,42,49,56,62,70,78,96,111,117,120/
      Data  nodes/ 2,2, 2,4, 0,4, 0,0, 4,0, 4,4, 2,4, 2,2,
     1             2,2, 2,4, 1,4, 0,3, 0,1, 1,0, 3,0, 4,1,
     &                  4,3, 3,4, 2,4, 2,2,
     2             2,2, 2,4, 0,1, 4,1, 2,4, 2,2,
     3             2,2, 4,2,-1,0, 2,4, 2,0,-1,0, 0,2, 2,2,
     4             2,2, 4,4,-1,0, 0,4, 4,0,-1,0, 0,0, 2,2,
     5             2,2, 2,4, 0,2, 2,0, 4,2, 2,4, 2,2, 
     6             2,2, 2,0, 2,4, 0,2, 4,2, 2,4, 2,2,
     7             2,2, 0,0, 4,4, 0,4, 4,0, 2,2,
     8             2,2, 4,4, 0,4, 4,4, 0,0, 4,0, 0,0, 2,2,
     9             2,2, 4,4,-1,0, 0,4, 2,2,-1,0, 2,0, 2,2,
     A             2,2, 4,4,-1,0, 3,3, 3,1, 4,0,-1,0, 0,0,
     &                  1,1, 1,3, 0,4,-1,0, 1,1, 3,1,-1,0,
     &                  1,3, 3,3, 2,2,
     1             2,2, 4,2,-1,0, 2,4, 2,0,-1,0, 0,2, 2,2,
     &                  4,4,-1,0, 0,4, 4,0,-1,0, 0,0, 2,2,
     2             2,2, 4,4, 0,4, 4,0, 0,0, 2,2,
     3             2,2, 2,4, 2,0/
C
      If ((itext .lt. 0) .or. (itext .gt. 13)) Then
         Return
      End If
C
      If (nc .lt. -1) Then
         ic = 2
      Else
         ic = 3
      End If
C
C...  Should current height and angle be used?
      If (hgt .le. 0.0) Then
         Goto 150
      End If
      fct = hgt / 4
C
C...  New angle in this call?
      If (angle .eq. theta) Then
C...     Is calculation of new offsets unnecessary?
         If (fct .eq. facc) Then
            Goto 150
         End If
         Goto 500
      End If
C
C...  Calculate a new THETA
      theta = angle
      ang = AMOD (angle,360.0)
      If (ang .lt. 0.0) Then
         ang = 360.0 + ang
      End If
      i = (ang + EPSIL) / 90.0
      a = i * 90.0
      If (ABS(ang-a) .le. EPSIL) Then
         ancs = asin(i+1)
         ancc = acos(i+1)
      Else
         ancc = theta * radco
         ancs = SIN (ancc)
         ancc = COS (ancc)
      End If
C
C...  Calculate offsets for new FACC and/or ANGLE
C
  500 facc = fct
      xi = facc * ancc
      yi = facc * ancs
      z = 0.0
      w = 0.0
      Do 510 j = 1,4
         z = z + xi
         w = w + yi
         xa(j) = z
  510    ya(j) = w
C
  150 xc = xz - xa(2) + ya(2)
      yc = yz - xa(2) - ya(2)
C
C...  Extract index into NODE array and node count
      index = khar(itext)
      ndknt = khar(itext+1) - index
C
C...  Extract the next node
  210 nodex = nodes(1,index)
      nodey = nodes(2,index)
      index = index + 1
C
C...  Check for pen up (NODEX = -1)
      If (nodex .lt. 0) Then
         ic = 3
         Goto 245
      End If
C
C...  Process move to node
      yt = yc + ya(nodex) + xa(nodey)
      xt = xc + xa(nodex) - ya(nodey)
      Call PLOT (xt,yt,ic)
      ic = 2
C
C...  Decrement and test node count
  245 ndknt = ndknt - 1
      If (ndknt .gt. 0) Then
         Goto 210
      End If
C
      Return
      End


