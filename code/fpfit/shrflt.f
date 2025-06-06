      subroutine shrflt(strike, dip, slip, tm)
c
c    this subroutine calculates the moment-tensor representation of a shear fault, given its strike, dip, and slip angles.
c
c  method:
c    the moment tensor is first expressed in a coordinate system with the z axis normal to the fault plane and the x axis in
c     the slip direction:
c
c                    (0.  0.  1.)
c                    (0.  0.  0.)
c                    (1.  0.  0.)
c
c    this coordinate system is then rotated through the euler angles phi = -slip, theta = -dip, and psi = strike - pi,
c    (conventions of goldstein, classical mechanics, sec 4-4) which results in a (south, east, up) orientation of the (x, y, z)
c    axes, respectively.  a permutation then converts this to the order (up, south, east).  the strength of the double-couple is
c    taken as unity; the calculated moment tensor components must be multiplied by the factor:
c
c         mu*a*s
c
c    where:
c         mu is the rigidity modulus of the medium
c         a is the fault area, and
c         s is the mean dislocation across the fault.
c           (note:  if the mean dislocation velocity is used instead,
c           the result will be the moment-rate tensor.)
c
c    written by bruce r. julian on 7 april, 1977.
c
      real              dip                             
c							! (input) fault dip angle in radians
      real              slip                            
c							! (input) fault slip angle in radians
      real              strike                          
c							! (input) fault strike angle in radians
      real              tm(6)                           
c							! (output) seismic moment tensor arranged in the following order:
                                                        
c							! (r, r)         i.e. (up, up)
                                                        
c							! (r, theta)     i.e. (up, south)
                                                        
c							! (theta, theta) i.e. (south, south)
                                                        
c							! (r, phi)       i.e. (up, east)
                                                        
c							! (theta, phi)   i.e. (south, east)
                                                        
c							! (phi, phi)     i.e. (east, east)
c
      real              a11                             
c							!  transformation matrix
      real              a21                             
c							!  transformation matrix
      real              a31                             
c							!  transformation matrix
      real              a13                             
c							!  transformation matrix
      real              a23                             
c							!  transformation matrix
      real              a33                             
c							!  transformation matrix
      real              cd                              
c							!  cos(dip)
      real              cl                              
c							!  cos(slip)
      real              cs                              
c							!  cos(strike)
      real              sd                              
c							!  sin(dip)
      real              sl                              
c							!  sin(slip)
      real              ss                              
c							!  sin(strike)
c
c  calculate components of orthogonal transformation matrix
c  from fault-oriented to (south, east, up) coordinate system
c
      ss = sin(strike)
      cs = cos(strike)
      sd = sin(dip)
      cd = cos(dip)
      sl = sin(slip)
      cl = cos(slip)
      a11 = -cs*cl - cd*sl*ss
      a21 =  ss*cl - cd*sl*cs
      a31 = sd*sl
      a13 = ss*sd
      a23 = cs*sd
      a33 = cd
c
c  transform moment tensor (0,   0,   1,
c                           0,   0,   0,
c                           1,   0,   0)
c
c  and permute axes to (up, south, east) order
c
      tm(1) = 2*a31*a33
      tm(2) = a11*a33 + a31*a13
      tm(3) = 2*a11*a13
      tm(4) = a21*a33 + a31*a23
      tm(5) = a11*a23 + a21*a13
      tm(6) = 2*a21*a23
c
      return
      end
