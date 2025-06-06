      logical function compl (solns, nsol, dd, da, sa, aerr,mxslns,naux)
c
c this function compares a "new" fault plane solution (dd, da, sa) with a list of other fault plane solutions
c  and checks for any of the following conditions
c
c   1. the "new" solution is similar to one of the solutions in solns
c   2. the compliment of the "new" solution is similar to one of the solutions in solns
c   3. the "new" solution is similar to the compliment of one of the solutions in solns
c
c if any one of the above conditions is true, function compl returns with a value .true.
c otherwise, function compl returns with the value .false.
c
c  solutions are similar if all three pairs of corresponding angles differ by less than aerr.
c
      real              aerr                            
c							! allowable difference between corresponding angles of complimentary planes
      real              da                              
c							! dip angle of new plane
      real              dd                              
c							! dip direction angle of new plane
      integer           mxslns                          
c							! maximum # of multiple solutions permitted
      integer           naux                            
c							! solns index of complimentary solution
      integer           nsol                            
c							! number of planes stored in array solns
      real              sa                              
c							! slip angle of new plane
      real              solns(mxslns,3)                 
c							! array of planes (dd, da, sa) to test new plane against
c
      real              rdiff                           
c							! function
      real              aux1(3)                         
c							! dip direction, angle, and rake of auxilliary plane of new plane
      real              aux2(3)                         
c							! dip direction, angle, and rake of auxilliary plane of prrevious planes
c
      compl = .false.
c
      call auxpln (dd, da, sa, aux1(1), aux1(2), aux1(3))
c
      do 10 naux = 1, nsol
c
c compare new solution with each solution on list
c
        if (abs(dd - solns(naux, 1)) .le. aerr .and.
     & abs(da - solns(naux, 2)) .le. aerr .and.
     & rdiff(sa, solns(naux, 3)) .le. aerr) then
          compl = .true.
          return
        end if
c
c     compare compliment of "new solution" with each solution on list
c
        if (abs(solns(naux, 1) - aux1(1)) .le. aerr .and.
     & abs(solns(naux, 2) - aux1(2)) .le. aerr .and.
     & rdiff(solns(naux, 3), aux1(3)) .le. aerr) then
          compl = .true.
          return
        end if
c
        call auxpln (solns(naux, 1), solns(naux, 2), solns(naux, 3),
     & aux2(1), aux2(2), aux2(3))
c
c     compare "new solution" with compliment of each solution on list
c
        if (abs(dd - aux2(1)) .le. aerr .and.
     & abs(da - aux2(2)) .le. aerr .and.
     & rdiff(sa, aux2(3)) .le. aerr) then
          compl = .true.
          return
        end if
10    continue
c
      return
      end
