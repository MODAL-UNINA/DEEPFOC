cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine MPAXIS (x,y,label,nchar,sizel,axlen,angle,VMIN,VMAX,dv,
     &                   ITIC,sizen,fmt,lcode)
C
C MPAXIS - Plot an annotated axis.
C
C Call MPAXIS (x,y,label,nchar,sizel,axlen,angle,vmin,vmax,dv,
C              itic,sizen,fmt,lcode)
C
C      (x,y)  = Starting coordinates for axis generation
C      label  = Alphanumeric text string for labelling the axis
C      nchar  = Number of characters in the axis label
C               + = Annotations are generated above the axis
C               - = Annotations are generated below the axis
C      sizel  = Character size of axis label
C      axlen  = Axis length in inches
C      angle  = Angle in degrees at which axis is to be drawn
C      vmin   = First axis value
C      vmax   = Last axis value
C      dv     = Delta annotation value
C      itic   = Tic labeling interval (1 = every one, 2 = every other, etc)
C      sizen  = Character size of numeric labels
C      fmt    = Fortran format for numeric labels
C      lcode  = 0 for labels parallel to axis, 1 for perpendicular
C               (lcode not checked yet, all labels parallel)
C
C    Called by:  User program
C
C        Calls:  PLOT, SYMBOL, NUMBER, IDEBLANK
C
C Commons used:  None
C
C-
C
      Parameter  (RADN = 0.01745329)
      Character*50 axnum,fmt*(*),label*(*)
      ticlen=.06
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
C      exp = 0.0
C      adv = ABS (dv)
C
C...  Zero delta annotation value?
C      If (adv .ne. 0.0) Then
C...     dv exponent calculation completed?
C   20    If (adv .ge. 99.0) Then
C            adv = adv / 10.0
C            exp = exp + 1.0
C            Goto 20
C         End If
C...     dv exponent calculation completed?
C   30    If (adv .lt. 0.01) Then
C            adv = adv * 10.0
C            exp = exp - 1.0
C            Goto 30
C         End If
C      End If
C
C...  Compute normalized fval and dv
C      val = fval * (10.0**(-exp))
C      adv = dv   * (10.0**(-exp))
C
      exp=0.0
      val=vmin
      adv=dv
C...  Set up angular orientation variables
      t2 = angle * RADN
      sina = SIN (t2)
      cosa = COS (t2)
      dlen = (axlen / abs(vmax-vmin))*abs(dv)
C
      dx = -sizen
      dy = (sizen/2+ticlen*2.+.05)*side - sizen/2
      xx = x + dx*cosa - dy*sina
      yy = y + dy*cosa + dx*sina
C
C...  Annotate axis
      ntic = int(abs(vmax-vmin)/abs(dv)+1)
      if ((itic.gt.0).and.(sizen.gt.0)) then
      Do 60 i = 1,ntic
         if (mod(i-1,itic).eq.0) then
            write(axnum,fmt)val
            nfmt=ideblank(axnum)
            call symbol(xx,yy,sizen,axnum,angle,nfmt)
            end if
         val = val + adv
         xx  = xx + cosa * dlen
   60    yy  = yy + sina * dlen
         end if
C
C...  Label axis
      t2 = nc
C
C...  Does dv exponent exist?
      If (exp .ne. 0.0) t2 = nc + 6
C
      dx = -(sizel/2)*t2 + 0.5*axlen
      dy = (sizel/2+sizen+ticlen*2.+.1)*side - sizel/2
      xx = x + dx*cosa - dy*sina
      yy = y + dy*cosa + dx*sina
      Call SYMBOL (xx,yy,sizel,label,angle,nc)
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
      dx = -ticlen * side * sina
      dy = +ticlen * side * cosa
      dxm = dx*2.0
      dym = dy*2.0
      xx = x
      yy = y
      Call PLOT (xx,yy,3)
      Do 80 i = 1,ntic
         Call PLOT (xx,yy,2)
         if (mod(i-1,itic).eq.0) then
         call plot (xx+dxm,yy+dym,2)
         else
         Call PLOT (xx+dx,yy+dy,2)
         end if
         Call PLOT (xx,yy,3)
         xx = xx + cosa * dlen
   80    yy = yy + sina * dlen
C
C...  Plot rest of axis
      xx = x + cosa * axlen
      yy = y + sina * axlen
      Call PLOT (xx,yy,2)
C
      Return
C
      End
