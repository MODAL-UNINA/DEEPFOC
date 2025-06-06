      real function rdiff (rake1, rake2)
c
c  returns with the smallest absolute difference in slip angle between rake1 and rake2.
c
c  rake convention follows aki & richards, 1980, quantitative seismology, p. 114
c
      real              rake1                           
c							! (input) first rake
      real              rake2                           
c							! (input) second rake
c
      real              a                               
c							! stores first rake
      real              b                               
c							! stores second rake
      real              c                               
c							! stores rake difference
c
      rdiff = 999.
      a = rake1
      if (rake1 .lt. 0.) a = 360. + rake1
      b = rake2
      if (rake2 .lt. 0.) b = 360. + rake2
      c = abs(a - b)
      if (c .gt. 180.) c = 360. - c
      rdiff = c
      return
      end
