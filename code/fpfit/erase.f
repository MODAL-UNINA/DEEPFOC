c
c sends a screen erase code to a tektronix terminal
c
      subroutine erase()
c                                                       ! ascii escape
      character a*1
c                                                       ! ascii form fee
cd
c
      character b*1
# 10 "erase.for"
      a = char(27)
      b = char(12)
      write(unit=6, fmt=10) a, b
c
# 13 "erase.for"
   10 format(1x,2a1)
# 15 "erase.for"
      return 
      end
