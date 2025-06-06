      subroutine askc (prompt, string)
c
c  askc prompts then reads a character string from the terminal.
c  the original value is unchanged by a cr response.
c
      character         prompt*(*)                      
c							! prompt string
      character         string*(*)                      
c							! character response, or original string on cr.

      character         temp*80                         
c							! scratch
      integer           leng                            
c							! function
      integer           nch                             
c							! number of characters
      integer		ounit
c							! logical unit for output 

      parameter (ounit = 0)
c     parameter (ounit = 6)						! VAX/VMS
      nch = leng(string)
10    write (ounit, 20) prompt
20    format (1x, a)
      if (nch .lt. 20) then
        write (ounit, 30) string(1:nch)
30      format (' [cr = ', a, ']? ', $)
      else
        write (ounit, 40) string(1:nch)
40      format (' [cr = ', a, ']?')
      end if
      read (5, '(a)', err = 10, end = 50) temp
      if (leng(temp) .gt. 0) string = temp
50    return
      end

