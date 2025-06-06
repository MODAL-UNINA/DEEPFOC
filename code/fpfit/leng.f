      integer function leng (string)
c
c the non-blank length of string whose physical length is maxlen
c (returns the position of the last non-blank character)
c
      character         string*(*)                      
c							! string
c
      integer           i                               
c							! character position
      integer           maxlen                          
c							! length of string

      maxlen = len(string)
      do 10 i = maxlen,1,-1
        if (string(i:i) .ne. ' ') goto 20
10    continue
      i = 0
20    leng = i
      return
      end
