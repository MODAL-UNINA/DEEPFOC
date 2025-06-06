      real function askr (prompt, dflt)
c
c  askr prompts then reads a real value from the terminal.
c  the default value is returned on a cr response.
c
      real              dflt                            
c							! default supplied on carriage return and displayed in prompt
      character         prompt*(*)                      
c							! prompt string

      integer           i                               
c							! loop index
      integer           j                               
c							! loop index
      integer           leng                            
c							! function
      character         temp*20                         
c							! scratch
      integer		ounit
c							! logical unit for output 

      parameter (ounit = 0)
c      parameter (ounit = 6)				! VAX/VMS version
      write (temp, 10) dflt
10    format (g20.5)
      do 20 i = 1, 20
        if (temp(i:i) .ne. ' ') goto 30
20    continue
30    do 40 j = 20, 1, -1
        if (temp(j:j) .ne. ' ') goto 50
40    continue
50    write (ounit, 60) prompt, temp(i:j)
60    format (1x, a, ' [cr = ', a, ']? ', $)
      read (5, '(a)', err = 50, end = 70) temp
      if (leng(temp) .gt. 0) then
        read (temp, *, err = 50) askr
      else
        askr = dflt
      end if
70    return
      end

