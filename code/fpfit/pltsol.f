	subroutine pltsol (dd, da, sa, pi, rmax1, cx1, cy1, hite)
c
c     Plot focal mechanism solution parameters
c                                                       ! ray azimuth
      real              cx1                             
c                                                       ! x position of large circle center
      real              cy1                             
c                                                       ! y position of large circle center
      real              da                             
c                                                       ! dip angle 
      real              dd                             
c                                                       ! strike
      real              hite                           
c                                                       ! height of string
      integer           i
c							! loop index
      integer           ida
c							! integer equivalent of da
      integer           idd
c							! integer equivalent of dd
      integer           isa
c							! integer equivalent of sa
      integer           ls
c							! left-shift index
      real              pi                              
c                                                       ! pi
      real              rmax1                           
c                                                       ! radius of large circle
      real              sa                             
c                                                       ! rake 
      character*80      string                           
c                                                       ! dummy string
      character*1       str
c                                                       ! dummy string
      real              xs                           
c                                                       ! x position of string
      real              ys                            
c                                                       ! y position of string

        dd = dd - 90.
        if (dd .lt. 0.) dd = dd + 360.
        idd = nint(dd)
        ida = nint(da)
        isa = nint(sa)
	write (string, 990) idd,ida,isa
990	format (i3, ',', i2, ',', i4) 
c
c Left shift string if blanks present
c
	ls = 11
	do 1000 i = 1, 10
995	  str = string(i:i)
	  if (str .eq. ' ') then
	    string(i:ls-1) = string(i+1:ls)
	    ls = ls - 1
	    goto 995
	  end if
1000	continue
c
c Find plot position
c
        xs = rmax1*cos((90. - dd)*pi/180.)
        ys = rmax1*sin((90. - dd)*pi/180.)
        if (xs .lt. 0.) then
          xs = cx1 + xs - hite*float(ls)
        else
          xs = cx1 + xs +.04*rmax1
        end if
        if (ys .lt. 0.) then
          ys = cy1 + ys - .04*rmax1 - hite
        else
          ys = cy1 + ys + .04*rmax1
        end if
	call symbol (xs, ys, hite, string, 0., ls)
c	call symbol (xs, ys, hite, %ref(string), 0., ls)		! VAX/VMS version
	return
	end
