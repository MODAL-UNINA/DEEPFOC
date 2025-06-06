      subroutine blurb (funit)
c
c writes preamble blurb to fit-function file
c
      integer           funit                           
c							! (input) logical unit # of output of fit listing for all strikes, dips
c
      integer           i                               
c							! loop index
c
      write (funit, 10) ('-', i = 1, 79)
10    format (79a1, /, 
     & 'note: the solution on the extended summary cards is expressed ',
     & 'in terms of', /, 
     & 'dip-direction, dip, and rake.  the fit-function tables can be ',
     & 'converted into', /,
     & 'this format by the following rules:', /,
     & '    if (dip > 90) then', /,
     & '       dip = 180 - dip', /,
     & '       strike = strike + 180', /,
     & '       rake = -rake')
      write (funit, 20) 
20    format ('    else if (dip < 0) then', /,
     & '       dip = - dip', /,
     & '       strike = strike + 180', /,
     & '       rake = rake + 180', /,
     & '    end if', /,
     & '    dip-direction = modulus(strike + 90, 360)', /,
     & '    if (dip-direction < 0) dip-direction = dip-direction + 360',
     & /, '    rake = modulus(rake, 360)', /,
     & '    if (rake > 180) rake = rake - 360', /,
     & '    if (rake < -180) rake = rake + 360')
      write (funit, 25) 
25    format ('    if (dip = 0) compute auxilliary plane', /,
     & '    if (dip = 90 and dip-direction >= 180) then', /,
     & '       rake = -rake', /,
     & '       dip direction = dip direction - 180', /,
     & '    end if')
      write (funit, 30) ('-', i = 1, 79)
30    format (/,
     & 'all fit scores are multiplied by 1000 and range from 0 to 999.',
     & '  fit scores', /, 
     & 'annotated with an "*" are <= the fit score of the function min',
     & 'ima (ie., the', /,
     & 'best solution) + 90% confidence increment.  fit scores annotat',
     & 'ed with an "a"', /,
     & 'denote function minima.  for coarse search, discrete relative ',
     & 'fit minima', /,
     & 'corresponding to multiple solutions are annotated with "b","c"',
     & ', etc.  tables', /,
     & 'of fit scores for fine search solutions corresponding to these',
     & ' multiples may', /,
     & 'not be shown if the multiple is recognized as the auxilliary s',
     & 'olution of', /,
     & 'another solution', /, 79a1)
      return
      end
