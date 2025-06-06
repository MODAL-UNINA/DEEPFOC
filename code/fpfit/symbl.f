cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine symbl (xz, yz, ht, itext, angle, nz)
c     modified from larry baker symbol.
c
c call symbl (x,y,hgt,itext,angle,nc)
c
c      (x,y)  = starting coordinate of the text generation.
c      ht     = character height specification (inches).
c               if hgt <= 0.0, slant letters are used.
c               (^'s in the slant string turn slant on and off...)
c      itext  = alphanumeric text to be generated.
c      angle  = angle at which the character line is plotted.
c      nc     = number of characters to be plotted.
c               negative means plot string backwards starting from right end.
c
c               >  0 =  alpha text, number of characters to be plotted.
c
c
c
      dimension  xs(25), ys(25), ipen(25)
      character itext*(*)
      dimension  asin(5), acos(5)
c
      common /symbol0/  xo,yo,xc,yc,theta,fct,xa,ya

      data  radco/0.01745329/,fnn/999.0/,fctr/0.7/,facc/0.0/,theta/0.0/
      data  ancc/1.0/,ancs/0.0/,xt/0.0/,yt/0.0/
      data  msk4/15/,msk5/31/,msk8/255/,msk11/2047/,mskall/-1/
      data  kbit/8/,mskbit/255/,nchar/2/,nchrs/128/,epsil/0.0000277/
      data  asin(1)/0./,asin(2)/1./,asin(3)/0./,asin(4)/-1./,asin(5)/0./
      data  acos(1)/1./,acos(2)/0./,acos(3)/-1./,acos(4)/0./,acos(5)/1./
c
      common/isavepi/ isaveflag, isave, isaveb

      x = xz
      y = yz
      hgt=ht
      nc = abs(nz)
      div = 6.0

c...  Check for positioning instructions.
c       Need to add rotations.
      if (itext(1:2).eq.'\\c') then
        itext = itext(3:)
        lent = lentrue (itext)
        xoff = 0.5*hgt*lent
        x = x - xoff
      else if (itext(1:2).eq.'\\r') then
        itext = itext(3:)
        lent = lentrue (itext)
        xoff = hgt*lent
        x = x - xoff
      endif

c...  character text output
      fct = abs(hgt)/div

c...  new angle in this symbol call?
      if (angle .ne. theta) then
c...  calculate a new theta
      theta = angle
      ang = amod(angle,360.0)
      if (ang .lt. 0) ang = 360.0 - ang
      i = (ang + epsil)/90.0
      a = i * 90.0
      if (abs(ang-a) .le. epsil) then
        ancs = asin(i+1)
        ancc = acos(i+1)
      else
        ancc = theta * radco
        ancs = sin(ancc)
        ancc = cos(ancc)
      endif
      endif

      xoff=0.
      if(nz.lt.0)  xoff=xoff-6.0
      yoff=0.

      do 260 ind=1,nc

      k=ind
c     plot string backwards if nz is negative...
      if(nz.lt.0) k=nc+1-ind

      if(itext(k:k).eq.' ') goto 250
      if(itext(k:k).eq.'^'.and.ht.lt.0.0) then
c        leave a little extra space to accomodate slant...
        if(hgt.lt.0.0.and.nz.gt.0) then
          xoff=xoff+3.0
        else if(hgt.gt.0.and.nz.lt.0) then
          xoff=xoff-3.0
        endif
        hgt=-hgt
        goto 260
      endif

      nvd = ichar(itext(k:k))
c      print *, 'nvd=',nvd

      call chargen(xs,ys,ipen,num,nvd,ichrl)
c      do 210 i=1,num
c 210  print *, xs(i),ys(i),ipen(i)
c     fix lower case 'i' to make dot more separate...
      if(nvd.eq.ichar('i')) then
        ys(2)=3.5
        ys(3)=5.5
      endif

      do 240 i=1,num
      xt=0.7*fct*(xs(i) + xoff)
      yt=fct*(ys(i) + yoff)

c     negative hgt signifies slant letters...
c      if(hgt.lt.0)  xt=xt+0.2*yt
c      if(hgt.lt.0)  xt=xt+0.25*yt
c      if(hgt.lt.0)  xt=xt+0.3*yt
      if(hgt.lt.0)  xt=xt+0.4*yt

      if(theta.ne.0.0) then
        xtp=(ancc*xt - ancs*yt)
        ytp=(ancs*xt + ancc*yt)
        xt=xtp
        yt=ytp
      endif

      xt = x + xt
      yt = y + yt
      ic=ipen(i)+2
c      print *, '>>',xt,yt,ic

      call nxplot (xt,yt,ic)
 240  continue

 250  xoff=xoff+sign(6.0,real(nz))
      yoff=yoff+0.

 260  continue

      return
      end


