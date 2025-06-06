      subroutine csort(cx, ix, n)
c
c  indirect sort routine from meissner & organick, p.352
c  stores ascending sort order of cx in array ix, leaving cx unchanged
c
      integer           ix(*)                           
c							! (output) pointer array to sorted order
      character*(*)     cx(*)                           
c							! (input) array to be sort
      integer           n                               
c							! (input) number of elements to be sorted
c
      integer           i                               
c							! loop index
      integer           j                               
c							! loop index
      integer           next                            
c							! index into cx
c
      do 10 i = 1, n
        ix(i) = i
10    continue
c
      do 40 j = 1, n - 1
        next = ix(j + 1)
        do 20 i = j, 1, -1
          if (cx(next) .gt. cx(ix(i))) goto 30
          ix(i + 1) = ix(i)
20      continue
30    ix(i + 1) = next
40    continue
c
      return
      end
