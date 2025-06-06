      subroutine intcas(imin,imax,n,icas)
      integer i,j,imin,imax,n,icas(n)
      real randn(n)
cf2py intent(in) imin,imax,n
cf2py intent(out) icas

       CALL RANDOM_SEED()
       i=1
10     CALL RANDOM_NUMBER(randn(i))
       randn(i) = imin + INT(randn(i) *
     &(imax - imin + 1))
          icas(i)=int(randn(i))
          do j=1,i-1
             if( int(randn(i)).eq.int(randn(j))) then
               i=i-1
               goto 10
             endif
          enddo
        if(i.eq.n) goto 20
        i=i+1
        goto 10
20      continue

       return
       end
