cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      integer function lentrue (string)
c     Gives position of last non-blank, non-tab, non-null character 
c     in a string.  Returns 0 if no such beast exists in the string.

      character*(*) string
      character*1 blank, tab, null
c      parameter (blank=' ', tab=char(9), null=char(0))

      lentrue=0
      tab = char(9)
      null = char(0)
      blank = ' '

      do 100 i=len(string),1,-1
        if (       string(i:i).ne.blank
     &       .and. string(i:i).ne.tab
     &       .and. string(i:i).ne.null) then
          lentrue=i
          return
        endif
 100  continue

      return
      end


