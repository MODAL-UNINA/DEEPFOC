      integer function jask (prompt, idflt)
c
c  jask prompts then reads an integer value from the terminal.
c  the default value is returned on a cr response.
c
      integer           idflt                           
c							! default supplied on carriage return and displayed in prompt
      character         prompt*(*)                      
c							! prompt string

      character         temp*20                         
c							! scratch
      integer           i                               
c							! loop index
      integer           leng                            
c							! function
      integer           ounit
c                                                       ! logical unit for output 

c      parameter (ounit = 0)
      parameter (ounit = 6)						! VAX/VMS
      write (temp, 10) idflt
10    format (i20)
      do 20 i = 1, 20
        if (temp(i:i) .ne. ' ') goto 30
20    continue
30    write (ounit, 40) prompt, temp(i:20)
40    format (1x, a, ' [cr = ', a, ']? ', $)
      read (5, '(a)', err = 30, end = 50) temp
      if (leng(temp) .gt. 0) then
        read (temp, *, err = 30) jask
      else
        jask = idflt
      end if
50    return
      end

