      logical function askl (prompt, dflt)
c
c  askl prompts then reads a logical value from the terminal.
c  the default value is returned on a cr response.
c
      logical           dflt                            
c							! default supplied on carriage return and displayed in prompt
      character         prompt*(*)                      
c							! prompt string
      integer           leng                            
c							! function
      character*7	temp
c							! scratch string
      integer		ounit
c							! logical unit for output (0 for UNIX, 6 for VMS)

      parameter (ounit = 6)
c      parameter (ounit = 0)
10    format (l1)
50    write (ounit, 60) prompt, dflt
60    format (1x, a, ' [cr = ', l2, ']? ', $)
      read (5, '(a)', err = 50, end = 70) temp
      if (leng(temp) .gt. 0) then
        read (temp, *, err = 50) askl
      else
        askl = dflt
      end if
70    return
      end

